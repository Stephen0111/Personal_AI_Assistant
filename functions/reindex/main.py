import functions_framework
import os
import tarfile
import shutil
from google.cloud import storage, secretmanager
from github import Github
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------------------------
# Memory / workload controls
# ----------------------------
MAX_REPOS = 10
MAX_FILES_PER_REPO = 8
MAX_FILE_SIZE_BYTES = 30000
BATCH_SIZE = 25
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_secret(name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    project = os.environ["GCP_PROJECT"]
    path = f"projects/{project}/secrets/{name}/versions/latest"
    response = client.access_secret_version(name=path)
    return response.payload.data.decode("utf-8")


def get_storage_bucket():
    bucket_name = os.environ["GCS_BUCKET"]
    return storage.Client().bucket(bucket_name)


def download_chroma_if_exists(bucket) -> bool:
    """
    Optional helper if you later want to download an existing archive first.
    Not currently used in this rebuild-from-scratch flow.
    """
    blob = bucket.blob("chroma_db.tar.gz")
    if blob.exists():
        blob.download_to_filename("/tmp/chroma_db.tar.gz")
        with tarfile.open("/tmp/chroma_db.tar.gz", "r:gz") as tar:
            tar.extractall("/tmp/")
        return True
    return False


def upload_chroma(bucket) -> None:
    tar_path = "/tmp/chroma_db.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("/tmp/chroma_db", arcname="chroma_db")
    bucket.blob("chroma_db.tar.gz").upload_from_filename(tar_path)


def fetch_repo_docs(repo, splitter: RecursiveCharacterTextSplitter):
    docs = []

    # Fetch README
    try:
        readme = repo.get_readme()
        readme_text = readme.decoded_content.decode(errors="ignore")
        chunks = splitter.create_documents(
            [readme_text],
            metadatas=[{"source": repo.full_name, "type": "readme"}]
        )
        docs.extend(chunks)
    except Exception as e:
        print(f"README skipped for {repo.full_name}: {e}")

    # Fetch selected text/code files
    try:
        contents = list(repo.get_contents(""))
        count = 0

        while contents and count < MAX_FILES_PER_REPO:
            file = contents.pop(0)

            if file.type == "dir":
                try:
                    contents.extend(repo.get_contents(file.path))
                except Exception as e:
                    print(f"Directory traversal skipped for {repo.full_name}/{file.path}: {e}")
                continue

            if not file.name.endswith((".py", ".md", ".txt")):
                continue

            if file.size > MAX_FILE_SIZE_BYTES:
                continue

            try:
                text = file.decoded_content.decode(errors="ignore")
                chunks = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "source": repo.full_name,
                        "type": "code",
                        "file": file.path
                    }]
                )
                docs.extend(chunks)
                count += 1
            except Exception as e:
                print(f"File skipped {repo.full_name}/{file.path}: {e}")

    except Exception as e:
        print(f"Repo traversal skipped for {repo.full_name}: {e}")

    return docs


@functions_framework.http
def reindex_all(request):
    print("Starting full re-index of all GitHub repos...")

    try:
        github_token = get_secret("github-token")
        gh = Github(github_token)

        # Get authenticated user's public non-fork repos
        user = gh.get_user()
        repos = [r for r in user.get_repos() if not r.fork and not r.private]
        repos = repos[:MAX_REPOS]

        print(f"Found {len(repos)} public non-fork repos to index (capped at {MAX_REPOS})")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        all_docs = []

        # Ingest GitHub repos
        for repo in repos:
            print(f"Fetching: {repo.full_name}")
            docs = fetch_repo_docs(repo, splitter)
            all_docs.extend(docs)
            print(f"  -> {len(docs)} chunks")

        # Ingest any docs in GCS /documents/ folder
        bucket = get_storage_bucket()
        blobs = bucket.list_blobs(prefix="documents/")
        for blob in blobs:
            if blob.name.endswith((".txt", ".md")):
                try:
                    text = blob.download_as_text()
                    chunks = splitter.create_documents(
                        [text],
                        metadatas=[{"source": blob.name, "type": "document"}]
                    )
                    all_docs.extend(chunks)
                    print(f"Ingested GCS doc: {blob.name} -> {len(chunks)} chunks")
                except Exception as e:
                    print(f"GCS document skipped {blob.name}: {e}")

        if not all_docs:
            print("No documents found. Exiting.")
            return {"status": "ok", "message": "No documents to index"}, 200

        print(f"Total chunks to embed: {len(all_docs)}")

        # Rebuild from scratch
        if os.path.exists("/tmp/chroma_db"):
            shutil.rmtree("/tmp/chroma_db")

        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma(
            persist_directory="/tmp/chroma_db",
            embedding_function=embedder
        )

        # Batch add to reduce memory spikes
        for i in range(0, len(all_docs), BATCH_SIZE):
            batch = all_docs[i:i + BATCH_SIZE]
            print(f"Embedding batch {i} to {i + len(batch)}")
            db.add_documents(batch)

        print("Persisting Chroma DB...")
        db.persist()

        print("Uploading Chroma archive to GCS...")
        upload_chroma(bucket)

        summary = {
            "status": "completed",
            "repos_indexed": len(repos),
            "chunks_indexed": len(all_docs),
            "batch_size": BATCH_SIZE
        }

        print(summary)
        return summary, 200

    except Exception as e:
        print(f"Reindex failed: {e}")
        return {"status": "error", "message": str(e)}, 500