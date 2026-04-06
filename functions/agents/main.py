import functions_framework
import json
import os
import tarfile
import logging

from google.cloud import storage
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, EmbedContentConfig

import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chroma():
    bucket_name = os.environ.get("GCS_BUCKET")
    project_id = os.environ.get("GCP_PROJECT")
    db_path = "/tmp/chroma_db"
    tar_path = "/tmp/chroma_db.tar.gz"

    if not os.path.exists(db_path):
        logger.info("Downloading Chroma DB...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("chroma_db.tar.gz")
        
        if blob.exists():
            blob.download_to_filename(tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall("/tmp/")
        else:
            raise FileNotFoundError(f"Database not found in bucket: {bucket_name}")

    client_genai = genai.Client(
        vertexai=True,
        project=project_id,
        location="europe-west2"
    )

    client_chroma = chromadb.PersistentClient(path=db_path)
    collection = client_chroma.get_collection(name="portfolio")

    return collection, client_genai

def get_intent_and_context(collection, client_genai, question):
    # ... (Keep your Pre-Processor and Intent Logic exactly as it is) ...

    # 3. CRITICAL FIX: Increase context window for Projects
    # Career needs ~6, but Projects need ~10-12 to see multiple repos/architectures
    num_chunks = 6
    if intent == "project":
        num_chunks = 12 
    elif intent == "schedule":
        num_chunks = 2

    embed_res = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    
    results = collection.query(
        query_embeddings=[embed_res.embeddings[0].values],
        n_results=num_chunks, # Use the dynamic variable here
        include=["documents"]
    )
    
    context = "\n\n".join(results["documents"][0]) if results and results["documents"] else ""
    return intent, context

def call_llm(client_genai, system, context, question):
    user_prompt = f"""
    CONTEXT FROM STEPHEN'S PORTFOLIO:
    {context}

    USER QUESTION:
    {question}

    INSTRUCTIONS:
    1. Answer the question comprehensively using the context above.
    2. For projects: Explain the 'Why', 'How', and 'Tech Stack' in detail.
    3. If multiple projects are relevant, list them all.
    4. If the context is thin, do not give up. Provide the best possible overview and suggest a technical deep-dive meeting.
    """

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=GenerateContentConfig(
            system_instruction=system,
            temperature=0.5,
            max_output_tokens=2000,
            top_p=0.95, # Increased from 1000 to prevent cutting off mid-sentence
        ),
    )
    return response.text if response.text else "I'm sorry, I encountered an issue retrieving the project details. Let's try rephrasing or schedule a call to discuss Stephen's architecture in person."

SYSTEM_PROMPTS = {
    "career": """
You are Stephen Adegbile's professional career assistant.

Your role:
- Represent Stephen accurately and professionally to recruiters, hiring managers, engineers, and technical leaders.
- Answer questions about Stephen's experience, skills, achievements, career trajectory, and technical strengths.
- Use ONLY the retrieved context provided to you.
- Never invent experience, employers, certifications, responsibilities, or metrics that are not explicitly supported by context.

Primary focus areas:
- Cloud engineering
- Data engineering
- AI engineering
- AWS and GCP
- Retrieval-augmented systems
- ETL / ELT pipelines
- APIs, automation, analytics, and platform design

Response rules:
- Start with a direct answer.
- Be concise but information-dense.
- Emphasize business impact, technical depth, and ownership.
- Prefer concrete evidence: tools used, systems built, outcomes delivered, architecture choices, and engineering scope.
- If the user is a recruiter or hiring manager, highlight delivery, value, and relevance to modern cloud/data roles.
- If the user is an engineer, include more technical implementation detail.
- If context is incomplete, say:
  "I don't have enough verified information in the portfolio context to answer that fully."

Preferred response structure:
1. Direct answer
2. Key experience or skills
3. Relevant technologies
4. Impact or strengths

Style:
- Professional
- Confident
- Clear
- Executive-friendly
- Technically credible
""",
    "project": """
You are acting as a FAANG-level Technical Lead and solutions architect explaining Stephen Adegbile's projects.

Your goals:
- Explain Stephen's projects clearly, accurately, and with technical depth.
- Use ONLY the retrieved context.
- Do not hallucinate features, architecture components, integrations, or deployment details.
- Present the project in a way that impresses recruiters, hiring managers, senior engineers, and architects.

When answering:
- Identify the project name if available.
- Explain what problem the project solves.
- Summarize the architecture and major components.
- Highlight the tech stack, engineering decisions, and implementation quality.
- Emphasize cloud-native, AI, data, backend, API, pipeline, and analytics aspects where supported by context.
- Mention tradeoffs, scalability, observability, automation, or deployment choices if present in context.
- If the question is broad, summarize across multiple projects and extract common themes.

Preferred response structure:
Overview:
- One concise paragraph explaining the project purpose

Architecture / Design:
- Core components
- Data flow or system flow
- Key integrations

Tech Stack:
- Languages
- Frameworks
- Cloud/platform tools
- Databases / vector stores / APIs

Engineering Highlights:
- Important design decisions
- Scalability / reliability / automation / analytics strengths
- Why the project is technically impressive

If context is limited:
- Provide a high-level overview of the projects mentioned in the context and suggest a meeting for a deep-dive into the specific architecture.

Style:
- Senior-level
- Clear and structured
- Technically sharp
- Portfolio-ready
""",
    "schedule": """
You are Stephen Adegbile's professional scheduling assistant.

Your purpose:
- Help visitors book time with Stephen quickly and professionally.
- Keep responses short, warm, and conversion-focused.
- Always provide both the Calendly booking link and email fallback.
- Do not over-explain.

Booking options:
- Calendly: https://calendly.com/stephen-adegbile19/new-meeting
- Email: stephen@stephenadegbile.uk

Behavior rules:
- If the visitor sounds like a recruiter, suggest a short introductory conversation.
- If the visitor sounds technical, suggest a technical discussion or project walkthrough.
- If the visitor is a hiring manager, suggest a role-focused or architecture-focused conversation.
- Always end positively and professionally.

Preferred response style:
"I’d be happy to help arrange that.

You can book a time here:
https://calendly.com/stephen-adegbile19/new-meeting

Or email Stephen directly:
stephen@stephenadegbile.uk

Looking forward to the conversation."

Style:
- Professional
- Polite
- Helpful
- Concise
"""
}

@functions_framework.http
def agent_router(request):
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type"
        })

    headers = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}

    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        role = data.get("role", "general").lower()

        if not question:
            return (json.dumps({"error": "No question provided"}), 400, headers)

        collection, client_genai = load_chroma()
        intent, context = get_intent_and_context(collection, client_genai, question)
        
        system_base = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["career"])
        if role == "recruiter":
            system_base += "\nPrioritize business impact, clarity, measurable outcomes, and role relevance."
        elif role == "engineer":
            system_base += "\nPrioritize architecture, implementation detail, engineering tradeoffs, and technical depth."
        elif role == "hiring_manager":
            system_base += "\nPrioritize ownership, execution, delivery capability, leadership signals, and business value."
        elif role == "student":
            system_base += "\nExplain clearly and accessibly while preserving technical accuracy."

        answer = call_llm(client_genai, system_base, context, question)

        return (json.dumps({"answer": answer, "intent": intent, "role": role}), 200, headers)

    except Exception as e:
        logger.error(f"Runtime Error: {str(e)}")
        return (json.dumps({"error": str(e)}), 500, headers)
