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
    q_lower = question.lower()
    
    # 1. Initialize intent with a default value to prevent 'NameError'
    intent = "career" 
    
    # 2. Intent Classification Logic
    if any(word in q_lower for word in ["project", "github", "repo", "built", "architecture"]):
        intent = "project"
    elif any(word in q_lower for word in ["book", "meeting", "calendly", "schedule", "yes", "ok"]):
        intent = "schedule"
    else:
        # LLM Fallback for intent
        intent_prompt = f"Classify this query into 'career', 'project', or 'schedule': {question}"
        try:
            intent_res = client_genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=intent_prompt,
                config=GenerateContentConfig(temperature=0, max_output_tokens=10)
            )
            llm_intent = intent_res.text.strip().lower() if intent_res.text else "career"
            if "project" in llm_intent: intent = "project"
            elif "schedule" in llm_intent: intent = "schedule"
            else: intent = "career"
        except Exception as e:
            logger.error(f"Intent LLM failed: {e}")
            intent = "career"

    # 3. Context Retrieval
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
        n_results=num_chunks,
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
            top_p=0.95,
        ),
    )
    return response.text if response.text else "I'm sorry, I encountered an issue. Let's try rephrasing or schedule a call."

SYSTEM_PROMPTS = {
    "career": """
You are Stephen Adegbile's professional career assistant.
Represent Stephen accurately. Answer questions about experience, skills, and achievements.
Use ONLY the retrieved context. Be professional and concise.
""",
    "project": """
You are a Technical Lead explaining Stephen Adegbile's projects.
Provide technical depth. Explain architecture, tech stack, and engineering decisions.
If context is limited, provide a high-level overview and suggest a meeting.
""",
    "schedule": """
You are Stephen Adegbile's scheduling assistant.
Always provide the Calendly link: https://calendly.com/stephen-adegbile19/new-meeting
And email: stephen@stephenadegbile.uk
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
        
        # Role-based prompt adjustment
        if role == "recruiter":
            system_base += "\nHighlight business impact and delivery."
        elif role == "engineer":
            system_base += "\nHighlight technical depth and architecture."

        answer = call_llm(client_genai, system_base, context, question)

        return (json.dumps({"answer": answer, "intent": intent, "role": role}), 200, headers)

    except Exception as e:
        logger.error(f"Runtime Error: {str(e)}")
        return (json.dumps({"error": str(e)}), 500, headers)
