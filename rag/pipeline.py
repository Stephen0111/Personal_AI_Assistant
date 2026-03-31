import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Free local embeddings — no API cost
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

CLIENT = chromadb.PersistentClient(path="./chroma_db")


def get_collection(name: str):
    return CLIENT.get_or_create_collection(
        name=name,
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_text(collection_name: str, text: str, doc_id: str, metadata: dict = None):
    """Chunk text and store embeddings in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    collection = get_collection(collection_name)
    collection.upsert(
        documents=chunks,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{**(metadata or {}), "source": doc_id}] * len(chunks),
    )
    print(f"Ingested {len(chunks)} chunks into '{collection_name}'")


def retrieve(collection_name: str, query: str, n_results: int = 4) -> list[str]:
    """Retrieve top-N relevant chunks for a query."""
    collection = get_collection(collection_name)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []
