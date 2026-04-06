import functions_framework
import json
import os
import tarfile
import logging
import time

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
    q = question.strip()
    q_lower = q.lower()

    # YOUR ORIGINAL DETAILED KEYWORD LISTS
    schedule_keywords = [
        "book", "booking", "schedule", "meeting", "meet", "call",
        "availability", "available", "calendar", "calendly",
        "intro chat", "intro call", "speak", "talk", "arrange time",
        "set up a meeting", "book a meeting", "book time", "contact"
    ]
    project_keywords = [
        "project", "projects", "built", "build", "architecture", "system",
        "pipeline", "pipelines", "platform", "app", "application",
        "rag", "assistant", "etl", "elt", "dashboard", "deployment",
        "solution", "case study", "implementation", "how it works",
        "github project", "github projects", "gcp project", "aws project",
        "portfolio project"
    ]
    career_keywords = [
        "experience", "skills", "skillset", "background", "career",
        "tech stack", "technology stack", "technology stacks", "stack",
        "technologies", "strengths", "expertise", "worked with", "roles",
        "role", "employment", "cv", "resume", "cloud engineering skills",
        "data engineering skills"
    ]
    short_schedule_confirmations = {"yes", "ok", "okay", "sure", "sounds good", "book it", "let's do it"}

    if len(q.split()) <= 3 and q_lower in short_schedule_confirmations:
        intent = "schedule"
    elif any(keyword in q_lower for keyword in schedule_keywords):
        intent = "schedule"
    elif any(keyword in q_lower for keyword in project_keywords):
        intent = "project"
    elif any(keyword in q_lower for keyword in career_keywords):
        intent = "career"
    else:
        intent = "career"

    if intent == "schedule" and len(q.split()) < 3:
        return intent, "The user has agreed or confirmed they want to schedule a meeting."

    # RESILIENCE: Retry loop for Embeddings
    embed_res = None
    for attempt in range(3):
        try:
            embed_res = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=[q],
                config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else: raise e

    n_val = 4 if intent == "project" else (2 if intent == "schedule" else 6)
    results = collection.query(
        query_embeddings=[embed_res.embeddings[0].values],
        n_results=n_val,
        include=["documents"]
    )

    context = "\n\n".join(results["documents"][0]) if results and results["documents"] else ""
    return intent, context

def call_llm(client_genai, system, context, question):
    user_prompt = f"Context:\n{context}\n\nUser Query:\n{question}\n\nInstructions: Use only the context above to answer. If scheduling or 'yes', provide booking details."

    # RESILIENCE: Retry loop for Gemini 2.5
    for attempt in range(3):
        try:
            response = client_genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    max_output_tokens=2000,
                ),
            )
            return response.text if response.text else "I couldn't generate a response."
        except Exception as e:
            if ("429" in str(e) or "404" in str(e)) and attempt < 2:
                time.sleep(2 ** attempt)
            else: raise e

# --- YOUR ORIGINAL FULL SYSTEM PROMPTS RESTORED ---
SYSTEM_PROMPTS = {
    "career": """
Role: Senior Talent Solutions Agent for Stephen Adegbile.
Objective: Provide high-signal, evidence-based responses to inquiries regarding Stephen’s professional background.

Operating Constraints:
- SOURCES: Use ONLY provided context. If data is missing, state: "My current portfolio context does not contain verified details on [X]."
- NO HALLUCINATION: Zero-tolerance for inventing metrics, years of experience, or specific tool proficiencies.
- TONE: Technically credible, executive-grade, and objective.

Communication Framework:
1. Direct Answer: 1-sentence executive summary.
2. Technical Evidence: Map skills to specific domains (Cloud, Data, AI).
3. Business Impact: Quantify success where context allows (e.g., "Reduced latency," "Automated ETL").
4. Adaptability:
   - If User = Recruiter/HM: Focus on 'Time-to-Value' and 'Production-Readiness'.
   - If User = Engineer: Focus on 'Stack Depth', 'Scalability', and 'Architecture Patterns'.

Required Response Structure:
- SUMMARY: Concise response to the prompt.
- CORE COMPETENCIES: Bulleted list of verified skills.
- STRATEGIC VALUE: How Stephen solves business problems using the above stack.
""",
    "project": """
Role: Principal Solutions Architect / Technical Lead.
Objective: Conduct a technical deep-dive into Stephen Adegbile's engineering portfolio.

Analysis Requirements:
- ARCHITECTURE FIRST: Prioritize the "How" and "Why" over a simple list of tools.
- SYSTEM DESIGN: Identify patterns (Event-driven, Microservices, RAG, Medallion Architecture).
- TRADEOFFS: Mention scalability, cost-efficiency, and data integrity decisions if context provides them.

Response Module Guidelines:
- Project Identity: Define the problem statement Stephen addressed.
- Stack Attribution: Explicitly link technologies (e.g., "Used AWS Kinesis for real-time ingestion").
- Engineering Excellence: Highlight automation (CI/CD), observability, or security implementations.

Structure:
## [Project Name]
- **Problem Statement:** Concise "Pain vs. Solution" description.
- **Architecture & Workflow:** Describe the data flow/logic path.
- **Tech Stack & Tooling:** Categorized list (e.g., Compute, Storage, AI/LLM).
- **Architectural Highlights:** Key engineering wins (e.g., "Serverless scaling," "Vector DB optimization").
- **Missing Context:** (Optional) If specific details like 'Deployment' or 'Testing' are absent, note them professionally.
""",
    "schedule": """
Role: Executive Scheduling Coordinator.
Objective: Convert interest into a confirmed calendar event with zero friction.

Execution Logic:
- KERNEL: The goal is a booking via Calendly (https://calendly.com/stephen-adegbile19/new-meeting).
- PERSONA-BASED ROUTING:
    - Recruiter: "Suggest a 15-min introductory sync."
    - Engineer: "Suggest a deep-dive technical walkthrough."
    - Hiring Manager: "Suggest a strategic fit/delivery discussion."

Constraint Checklist:
- Include Calendly Link: MANDATORY.
- Include Email (stephen@stephenadegbile.uk): MANDATORY.
- Length: Keep under 60 words.
- Tone: Warm, professional, and biased toward action.

Response Template:
"I can certainly facilitate that. Based on your interest, Stephen is available for a [sync/walkthrough].

**Book via Calendly:** https://calendly.com/stephen-adegbile19/new-meeting
**Direct Email:** stephen@stephenadegbile.uk

Looking forward to connecting."
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
        
        # YOUR ORIGINAL MODIFIERS
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
