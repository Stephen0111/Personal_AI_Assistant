import os
from github import Github
from dotenv import load_dotenv
from rag.pipeline import ingest_text

load_dotenv()

g = Github(os.getenv("GITHUB_TOKEN"))


def ingest_all_repos():
    """Pull README from every public repo and embed it."""
    user = g.get_user(os.getenv("GITHUB_USERNAME"))
    for repo in user.get_repos():
        if repo.private:
            continue
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode("utf-8")
            ingest_text(
                collection_name="github_projects",
                text=content,
                doc_id=repo.name,
                metadata={"repo": repo.name, "url": repo.html_url},
            )
        except Exception as e:
            print(f"Skipped {repo.name}: {e}")


def ingest_single_repo(repo_name: str):
    """Called by webhook when a new repo is created."""
    user = g.get_user(os.getenv("GITHUB_USERNAME"))
    repo = user.get_repo(repo_name)
    try:
        readme = repo.get_readme()
        content = readme.decoded_content.decode("utf-8")
        ingest_text(
            collection_name="github_projects",
            text=content,
            doc_id=repo.name,
            metadata={"repo": repo.name, "url": repo.html_url},
        )
    except Exception as e:
        print(f"Failed to ingest {repo_name}: {e}")


if __name__ == "__main__":
    # Ingest your CV once manually
    with open("data/cv.txt") as f:
        ingest_text("career", f.read(), "cv", {"type": "cv"})
    # Then ingest all repos
    ingest_all_repos()
    print("Initial ingestion complete.")
