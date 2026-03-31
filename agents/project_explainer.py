from rag.pipeline import retrieve
from agents.controller import call_llm


def answer(query: str) -> str:
    context_chunks = retrieve("github_projects", query, n_results=5)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a technical assistant explaining software projects.
Use the project documentation below to answer the question. Be specific.

Project documentation:
{context}

Question: {query}

Answer:"""
    return call_llm(prompt)
