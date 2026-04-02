import functions_framework
import json
import os
import tarfile

from google.cloud import storage
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_chroma():
    bucket = storage.Client().bucket(os.environ["GCS_BUCKET"])
    blob = bucket.blob("chroma_db.tar.gz")

    if not os.path.exists("/tmp/chroma_db") and blob.exists():
        blob.download_to_filename("/tmp/chroma_db.tar.gz")
        with tarfile.open("/tmp/chroma_db.tar.gz", "r:gz") as tar:
            tar.extractall("/tmp/")

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="/tmp/chroma_db",
        embedding_function=embedder
    )


def classify_intent(question):
    keywords = {
        "career": ["experience", "job", "worked", "cv", "resume", "skills", "career"],
        "project": ["project", "github", "built", "code", "architecture", "how does"],
        "schedule": ["meeting", "book", "call", "availability", "calendly", "schedule"],
    }

    q = (question or "").lower()
    for intent, words in keywords.items():
        if any(word in q for word in words):
            return intent
    return "career"


def retrieve_context(db, question, k=4):
    docs = db.similarity_search(question, k=k)
    return "\n\n".join(d.page_content for d in docs)


def call_llm(system, context, question):
    project_id = os.environ["GCP_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )

    user_prompt = f"""Context:
{context}

Question:
{question}

Instructions:
- Answer using only the context above.
- If the context is insufficient, say so clearly.
- Do not invent facts.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=GenerateContentConfig(
            system_instruction=[system],
            temperature=0.2,
            max_output_tokens=600,
        ),
    )

    return response.text if response.text else "Sorry, I could not generate a response."


SYSTEM_PROMPTS = {
    "career": """
You are an AI career assistant representing the portfolio owner.

Your goals:
- Answer recruiter and hiring manager questions about experience
- Highlight impact, results, and technologies used
- Be concise, confident, and professional
- Use ONLY the provided context
- Never invent experience not present in context

Response guidelines:
- Start with a direct answer
- Use bullet points where helpful
- Emphasize:
  - Technologies used
  - Business impact
  - Scale of work
  - Cloud / AI / data engineering skills
- If context is missing, say:
  "I don't have enough information in the portfolio to answer that."

Tone:
Professional, recruiter-friendly, confident, concise.
""",
    "project": """
You are a senior AI engineer explaining portfolio projects.

Your goals:
- Explain projects clearly and technically
- Highlight architecture decisions
- Emphasize engineering depth
- Use ONLY provided context
- Do not hallucinate missing components

Structure responses like:

Overview:
Brief explanation of the project

Architecture:
Describe system components

Tech Stack:
List technologies used

Key Features:
Bullet points

Engineering Highlights:
Why the project is impressive

Tone:
Technical, clear, concise, senior-level.

Focus on:
- Cloud architecture
- AI / RAG pipelines
- Data engineering
- APIs
- scalability
- deployment

Avoid:
- generic explanations
- vague descriptions
- repetition
""",
    "schedule": """
You are a scheduling assistant helping visitors book a meeting.

Your goals:
- Encourage booking a meeting
- Be friendly and professional
- Provide both booking link and email
- Tailor response based on intent

Instructions:
- If user is recruiter: suggest short intro call
- If engineer: suggest technical deep dive
- If hiring manager: suggest architecture discussion

Always include:

Calendly booking link:
https://calendly.com/stephen-adegbile19/new-meeting

Email fallback:
stephen@stephenadegbile.uk

Example response style:

"I'd be happy to discuss this further.

You can book a time directly here:
https://calendly.com/stephen-adegbile19/new-meeting

Or email me:
stephen@stephenadegbile.uk

Looking forward to speaking with you."
""",
}


@functions_framework.http
def agent_router(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)

    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        role = data.get("role", "general").strip().lower()

        if not question:
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json",
            }
            return (
                json.dumps({"error": "question is required"}),
                400,
                headers,
            )

        intent = classify_intent(question)
        db = load_chroma()
        context = retrieve_context(db, question)
        system = SYSTEM_PROMPTS[intent]

        if role == "recruiter":
            system += "\nKeep answers brief and highlight business impact and measurable outcomes."
        elif role == "engineer":
            system += "\nInclude technical depth, architecture choices, and implementation details."
        elif role == "hiring manager":
            system += "\nEmphasize ownership, delivery, leadership, and business value."
        elif role == "student":
            system += "\nExplain clearly, simply, and in an educational way."

        answer = call_llm(system, context, question)

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
        }
        return (
            json.dumps(
                {
                    "answer": answer,
                    "intent": intent,
                    "role": role,
                }
            ),
            200,
            headers,
        )

    except Exception as e:
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
        }
        return (
            json.dumps({"error": str(e)}),
            500,
            headers,
        )