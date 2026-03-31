from rag.pipeline import retrieve
from agents.controller import call_llm


def answer(query: str, role: str = "general") -> str:
    context_chunks = retrieve("career", query, n_results=4)
    context = "\n\n".join(context_chunks)

    role_instruction = {
        "recruiter": "Focus on job titles, years of experience, and key achievements.",
        "engineer": "Go into technical depth about tools, architecture, and code.",
        "hiring_manager": "Highlight leadership, project ownership, and business impact.",
    }.get(role, "Be clear and concise.")

    prompt = f"""You are a personal AI assistant representing this person's career.
{role_instruction}

Context from their CV and profile:
{context}

Question: {query}

Answer:"""
    return call_llm(prompt)
