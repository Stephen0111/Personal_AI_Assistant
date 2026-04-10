import functions_framework
import json
import logging
import os
import re
import tarfile
import threading
import uuid
from pathlib import Path
from typing import Literal, Optional, TypedDict

from google.cloud import storage
from google import genai
from google.genai.types import EmbedContentConfig

import chromadb
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy
from langchain_google_vertexai import ChatVertexAI

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Globals / singleton resources
# -----------------------------------------------------------------------------

_RESOURCE_LOCK = threading.Lock()
_RESOURCES = None
_GRAPH = None
_GRAPH_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_RETRIEVAL_DOCS = 14
CAREER_PER_QUERY_RESULTS = 4
GENERAL_CAREER_QUERY_LIMIT = 8
JOB_FIT_QUERY_LIMIT = 6
JOB_FIT_GENERAL_DOCS = 8
JOB_FIT_BUCKET_DOCS = 2

FIT_SIGNAL_PATTERNS = [
    r"\bdo you qualify\b",
    r"\bdo you meet\b",
    r"\bmeet the requirement",
    r"\bmeet the requirements\b",
    r"\bgood fit\b",
    r"\bstrong fit\b",
    r"\bfit for this role\b",
    r"\bfit for this job\b",
    r"\bsuitable for\b",
    r"\beligible for\b",
    r"\bmatch this role\b",
    r"\bmatch this job\b",
    r"\bjob description\b",
    r"\bperson specification\b",
    r"\bessential requirements\b",
    r"\bhighly desirable\b",
    r"\bqualifications\b",
    r"\brequirements\b",
]

REQUIREMENT_ALIASES = {
    "aws": ["aws", "amazon web services", "bedrock", "sagemaker", "glue", "emr", "s3", "redshift", "mwaa", "kinesis", "lambda", "athena"],
    "gcp": ["gcp", "google cloud", "vertex ai", "bigquery", "cloud run", "cloud composer", "dataflow", "gcs", "cloud functions", "pub/sub", "pubsub"],
    "azure": ["azure", "azure openai", "azure ai", "azure machine learning", "azure ml", "microsoft fabric", "azure data factory", "azure app service"],
    "databricks": ["databricks", "delta lake", "delta live tables", "unity catalog"],
    "snowflake": ["snowflake", "snowpipe"],
    "terraform": ["terraform"],
    "airflow": ["airflow", "cloud composer", "mwaa"],
    "python": ["python", "fastapi"],
    "react": ["react", "typescript", "node.js", "express"],
    "rag": ["rag", "retrieval-augmented generation", "vector database", "vector db", "pinecone", "chromadb", "langchain"],
    "multi_agent": ["agentic", "multi-agent", "multi agent", "orchestration", "tool use"],
    "security": ["security", "secure", "data protection", "compliance", "governance", "audit", "responsible ai"],
    "stakeholder": ["stakeholder", "client", "presentation", "review board", "pre-sales", "presales", "communication"],
    "software": ["software engineering", "api", "microservices", "event-driven", "devops", "ci/cd", "container", "docker"],
    "finance": ["financial services", "bank", "banking", "fintech", "trading", "regulatory"],
    "government": ["government", "public sector", "department"],
}

# -----------------------------------------------------------------------------
# State schema
# -----------------------------------------------------------------------------

class ConversationTurn(TypedDict):
    role: str
    content: str


class AgentState(TypedDict, total=False):
    question: str
    role: str
    user_type: str
    thread_id: str
    conversation_history: list[ConversationTurn]
    intent: Literal["career", "project", "schedule"]
    retrieved_context: str
    retrieved_docs: list[str]
    retrieval_queries: list[str]
    is_job_fit: bool
    structured_response: dict
    refined_response: str
    grounding_notes: str
    needs_refine: bool
    answer: str


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class RouteDecision(BaseModel):
    intent: Literal["career", "project", "schedule"] = Field(
        description="Best semantic route for the user's question."
    )
    inferred_user_type: Literal["recruiter", "engineer", "hiring_manager", "student", "general"] = Field(
        description="Inferred user type from wording and intent."
    )


class CareerResponse(BaseModel):
    summary: str = Field(description="One-sentence executive summary.")
    core_competencies: list[str] = Field(description="Verified competencies grounded in context.")
    strategic_value: str = Field(description="How Stephen creates business value or fit assessment.")


class ProjectResponse(BaseModel):
    project_name: str = Field(description="Name of the project being discussed.")
    problem_statement: str = Field(description="Pain vs solution summary.")
    architecture_workflow: str = Field(description="High-level system/data flow.")
    tech_stack_tooling: list[str] = Field(description="Relevant technologies grouped or listed.")
    architectural_highlights: list[str] = Field(description="Key engineering wins.")
    missing_context: Optional[str] = Field(default=None, description="Any important missing context.")


class ScheduleResponse(BaseModel):
    summary: str = Field(description="Short booking-oriented sentence.")
    competencies: list[str] = Field(description="Small list reflecting meeting type or agenda.")
    strategic_value: str = Field(description="Why the meeting would be useful.")
    booking_link: str = Field(description="Calendly URL.")
    direct_email: str = Field(description="Direct email address.")


class GroundingCheck(BaseModel):
    grounded: bool = Field(description="Whether the draft is fully supported by retrieved context.")
    revised_answer: Optional[str] = Field(
        default=None,
        description="If not grounded, return a corrected answer that preserves required formatting."
    )
    notes: str = Field(description="Brief explanation of what was changed or verified.")


class RetrievalPlan(BaseModel):
    is_job_fit: bool = Field(description="Whether the question is a qualification or role-fit assessment.")
    condensed_focus: str = Field(description="Short retrieval focus phrased as evidence to retrieve about Stephen.")
    priority_requirements: list[str] = Field(description="Top grouped requirements distilled from the role or user ask.")
    evidence_queries: list[str] = Field(description="Short search queries aimed at retrieving Stephen's evidence.")


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the semantic router for Stephen Adegbile's portfolio assistant.

Your only job is to choose the best destination for the latest user message.

Available destinations:
- career: questions about Stephen's background, CV, experience, strengths, skills, technologies, seniority, suitability, eligibility, role match, qualification, job-fit analysis, requirement matching, resume-to-job-description comparison, interview-fit assessment
- project: project deep-dives, architecture, implementation, system design, technical walkthroughs of a specific build, portfolio case studies
- schedule: booking, meetings, availability, intro calls, next steps, arranging time

Critical routing rules:
1. If the user asks whether Stephen/you qualify, fit, meet requirements, are suitable, are eligible, are a strong match, or are a good fit for a role, ALWAYS route to career.
2. If the user pastes a job description, person specification, role requirements, qualifications, responsibilities, essential/desirable criteria, or asks for a match assessment against a role, ALWAYS route to career.
3. Even if the pasted job description contains project, architecture, AI, cloud, or engineering terms, the route is still career when the user's goal is evaluating fit for the role.
4. Route to project only when the user wants a technical explanation of a specific project, architecture, implementation, or system.
5. Route to schedule only when the user clearly wants to arrange time, book a meeting, discuss availability, or next-step coordination.

Infer the user's role if possible: recruiter, engineer, hiring_manager, student, or general.
Return structured output only."""),
    ("human", "Conversation history:\n{conversation_history}\n\nLatest user question:\n{question}\n\nCurrent stated role from request:\n{role}\n"),
])

RETRIEVAL_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval planning specialist for Stephen Adegbile's portfolio assistant.

Your task is to convert the latest user question into high-quality retrieval queries that find evidence ABOUT STEPHEN from his CV/project corpus.

Rules:
- Do not assume any skill exists unless retrieved from the corpus.
- Do not insert technologies as implied strengths.
- If the user pasted a long job description, distill it into the most important grouped requirements.
- Evidence queries must be short, targeted, and framed around Stephen's experience.
- Prefer 4 to 6 evidence queries.
- Focus on retrieving evidence, not judging fit.
- Do not write long prose.
- Do not restate the entire JD.

Return structured output only."""),
    ("human", "Conversation history:\n{conversation_history}\n\nLatest question:\n{question}\n"),
])

CAREER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Role: Senior Talent Intelligence and Hiring Assessment Agent for Stephen Adegbile.

Primary objective:
Answer questions about Stephen's background, skills, experience, technologies, seniority, and fit for jobs using ONLY the provided context.

Non-negotiable rules:
- Use ONLY the retrieved context.
- Do NOT invent experience, certifications, employers, dates, responsibilities, domain exposure, or technologies.
- If the context does not verify something, say so explicitly.
- Do NOT overclaim. Be evidence-based and precise.
- Treat pasted job descriptions, person specifications, and role requirements as part of the user's question to be analysed.
- When the user asks "Do you qualify?", "Are you a good fit?", "Do you meet the requirements?", or similar, perform a requirement-match assessment grounded only in the context.

Important behavior for long job descriptions:
- Identify the most important requirements first.
- Group overlapping requirements together.
- Prioritize essentials over nice-to-haves.
- Produce a concise but high-signal answer that remains grounded.

Interpretation rules:
- If the user's message includes a job description or role requirements, your task is NOT to summarize Stephen generally.
- Your task is to compare Stephen's verified background against the role and answer whether he appears to be:
  - strong fit
  - reasonable / partial fit
  - weak or unverified fit
- Base that conclusion only on available evidence.
- If evidence is mixed, say so.

- For non-job-fit career questions such as tech stack, data engineering skills, cloud experience, AI skills, or software engineering background:
  - provide a richer and more comprehensive technical breakdown
  - group technologies into meaningful categories where appropriate
  - mention specific tools, platforms, frameworks, services, orchestration tools, data platforms, and engineering patterns that are explicitly supported by the retrieved context
  - prefer breadth plus specificity over overly compressed summaries
  - when cloud or data questions are asked, include named services and tools such as storage layers, ETL engines, orchestration frameworks, lakehouse/governance components, warehouses, APIs, and deployment tooling whenever they are supported by context
  - when multiple cloud platforms are supported by context, mention them explicitly rather than collapsing them into a single generic cloud label

Response design rules:
- Adapt depth by user type:
  recruiter -> emphasize requirement match, commercial relevance, time-to-value, production readiness, seniority signals
  engineer -> emphasize technical alignment, platforms, architecture depth, implementation credibility
  hiring_manager -> emphasize ownership, delivery scope, stakeholder impact, role fit, gaps/risk areas
  student -> explain clearly and simply

- For job-fit / qualification / suitability questions, the summary MUST begin exactly with one of:
  - VERDICT: Yes —
  - VERDICT: Partial match —
  - VERDICT: No —

- Use VERDICT: Yes — only when the retrieved context strongly supports the match.
- Use VERDICT: Partial match — when there is meaningful alignment but also material gaps or unverified requirements.
- Use VERDICT: No — when the retrieved context does not support key requirements.

Output requirements:
You must produce:
1) summary
2) core_competencies
3) strategic_value

Additional formatting guidance:
- In summary, give the direct answer first.
- In core_competencies, include only verified evidence-backed competencies or role-match points.
- For non-job-fit questions, core_competencies should be detailed and can contain a larger number of specific items when strongly supported by context.
- In strategic_value, explain how Stephen would create value OR explain the fit/gap assessment in a grounded way.
- If important requirements are unverified, say that clearly in the summary or strategic_value."""),
    ("human", "User type: {user_type}\nConversation history:\n{conversation_history}\n\nRetrieved context:\n{context}\n\nQuestion:\n{question}\n\nIs job-fit question: {is_job_fit}\n"),
])

PROJECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Role: Principal Solutions Architect / Technical Lead.

Rules:
- Use ONLY the retrieved context.
- Prioritize architecture, workflow, and engineering tradeoffs.
- Mention missing details when not present.
- Adapt depth by user type:
  recruiter -> explain impact and relevance
  engineer -> explain architecture, patterns, implementation choices
  hiring_manager -> explain scope, ownership, delivery value
  student -> explain clearly with structure

You must produce:
project_name, problem_statement, architecture_workflow, tech_stack_tooling,
architectural_highlights, missing_context"""),
    ("human", "User type: {user_type}\nConversation history:\n{conversation_history}\n\nRetrieved context:\n{context}\n\nQuestion:\n{question}\n"),
])

SCHEDULE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Role: Executive Scheduling Coordinator.

Rules:
- Keep concise and action-oriented.
- Include booking link and direct email.
- Adapt meeting framing by user type:
  recruiter -> 15-min introductory sync
  engineer -> technical walkthrough
  hiring_manager -> strategic fit / delivery discussion
  general -> intro conversation
- Use structured output only."""),
    ("human", "User type: {user_type}\nQuestion:\n{question}\n\nBooking link: https://calendly.com/stephen-adegbile19/new-meeting\nDirect email: stephen@stephenadegbile.uk\n"),
])

REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict grounding and hallucination checker.

Your job:
- Compare the candidate answer against the retrieved context.
- If unsupported claims appear, mark grounded=false and rewrite the answer using only supported facts.
- If the answer is supported, keep it.
- Be strict.

Critical formatting rule:
- Preserve the original answer structure unless a correction is necessary.
- If a correction is necessary, preserve the same final section headings and ordering.

For job-fit / qualification / suitability questions:
- The revised answer MUST preserve this exact structure:

SUMMARY:
VERDICT: Yes — ...   OR
VERDICT: Partial match — ...   OR
VERDICT: No — ...

CORE COMPETENCIES:
- ...
- ...

STRATEGIC VALUE:
...

- Do not rewrite job-fit answers into generic prose.
- Do not replace the section headings with new headings.
- Do not remove the verdict.
- Only downgrade claims that are unsupported.
- If some requirements are supported and others are missing or unclear, use VERDICT: Partial match —
- Do not permit invented years of experience, invented cloud/platform depth, invented stakeholder exposure, invented architecture leadership, invented governance depth, or invented client delivery.
- Prefer cautious, evidence-based wording when support is incomplete.

For non-job-fit career answers:
- Preserve useful technical detail and category structure when the content is supported by the retrieved context.
- Do not unnecessarily compress a detailed technical answer into a shorter generic summary.
- Preserve specific named technologies, services, and tools when they are supported by the retrieved context."""),
    ("human", "Is job-fit question: {is_job_fit}\n\nRetrieved context:\n{context}\n\nCandidate answer:\n{candidate_answer}\n"),
])

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _format_history(history: list[ConversationTurn]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-8:])


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_job_fit_question(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(pattern, q) for pattern in FIT_SIGNAL_PATTERNS)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        clean = _normalize_whitespace(item)
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def _extract_requirement_keys(text: str) -> list[str]:
    q = (text or "").lower()
    return [key for key, aliases in REQUIREMENT_ALIASES.items() if any(alias in q for alias in aliases)]


def _skill_query_from_key(key: str) -> str:
    mapping = {
        "aws": "Stephen Adegbile AWS experience cloud delivery architecture",
        "gcp": "Stephen Adegbile GCP experience cloud delivery architecture",
        "azure": "Stephen Adegbile Azure experience",
        "databricks": "Stephen Adegbile Databricks Delta Lake Unity Catalog Delta Live Tables delivery",
        "snowflake": "Stephen Adegbile Snowflake Snowpipe dbt data warehouse delivery",
        "terraform": "Stephen Adegbile Terraform infrastructure as code delivery",
        "airflow": "Stephen Adegbile Apache Airflow MWAA Cloud Composer orchestration delivery",
        "python": "Stephen Adegbile Python FastAPI backend engineering",
        "react": "Stephen Adegbile React TypeScript frontend engineering",
        "rag": "Stephen Adegbile RAG vector database retrieval system delivery",
        "multi_agent": "Stephen Adegbile multi-agent agentic workflow orchestration",
        "security": "Stephen Adegbile security compliance governance delivery",
        "stakeholder": "Stephen Adegbile stakeholder client communication presentations",
        "software": "Stephen Adegbile software engineering APIs microservices DevOps",
        "finance": "Stephen Adegbile financial services fintech delivery",
        "government": "Stephen Adegbile government public sector delivery",
    }
    return mapping[key]


def _question_mentions_any(question: str, keywords: list[str]) -> bool:
    q = (question or "").lower()
    return any(k in q for k in keywords)


def _build_general_career_queries(question: str) -> list[str]:
    q = (question or "").lower()

    base_queries = [
        "Stephen Adegbile CV profile technical skills experience",
        "Stephen Adegbile projects architecture tools platforms technologies",
    ]

    tech_stack_queries = [
        "Stephen Adegbile tech stack tools technologies languages frameworks platforms",
        "Stephen Adegbile cloud platforms AWS GCP Azure services deployment architecture",
        "Stephen Adegbile data engineering stack Spark Glue EMR Dataflow Airflow Composer MWAA Databricks Snowflake dbt BigQuery Redshift Athena",
        "Stephen Adegbile AI stack Vertex AI Gemini Hugging Face LangChain RAG ChromaDB Pinecone multi-agent systems",
        "Stephen Adegbile backend frontend FastAPI React Node Express PHP Laravel Streamlit APIs",
        "Stephen Adegbile DevOps Terraform Docker GitHub Actions CI CD cloud functions cloud run",
    ]

    cloud_queries = [
        "Stephen Adegbile cloud engineering AWS S3 Glue EMR MWAA Kinesis Lambda Redshift Athena SageMaker",
        "Stephen Adegbile cloud engineering GCP Cloud Functions Gen2 GCS BigQuery Vertex AI Cloud Composer PubSub Cloud Run",
        "Stephen Adegbile cloud architecture Databricks Snowflake Terraform Docker GitHub Actions CI CD",
        "Stephen Adegbile object storage data lake lakehouse medallion architecture S3 GCS Delta Lake",
        "Stephen Adegbile orchestration MWAA Cloud Composer Databricks Workflows Airflow",
    ]

    data_eng_queries = [
        "Stephen Adegbile data engineering PySpark Apache Spark AWS Glue EMR Google Cloud Dataflow",
        "Stephen Adegbile orchestration Apache Airflow MWAA Cloud Composer Databricks Workflows",
        "Stephen Adegbile lakehouse warehouse Databricks Delta Lake Delta Live Tables Unity Catalog Snowflake BigQuery Redshift Athena dbt",
        "Stephen Adegbile data quality governance medallion architecture SCD Type 2 schema enforcement Delta Live Tables Expectations Unity Catalog",
        "Stephen Adegbile data pipelines S3 parquet GCS BigQuery Snowpipe dbt analytics ready datasets",
    ]

    ai_eng_queries = [
        "Stephen Adegbile AI engineering Vertex AI Gemini Hugging Face LangChain RAG ChromaDB Pinecone multi-agent systems",
        "Stephen Adegbile AI projects personal AI assistant quant research platform AI test case generator startup valuation AlphaGoal",
        "Stephen Adegbile model serving FastAPI MLOps predictive analytics vector databases embeddings",
        "Stephen Adegbile production AI systems cloud functions streamlit event driven workflows webhook orchestration",
    ]

    software_queries = [
        "Stephen Adegbile software engineering FastAPI React Node Express PHP Laravel JavaScript TypeScript APIs",
        "Stephen Adegbile frontend backend full stack Streamlit React Bootstrap Tailwind PostgreSQL MySQL MongoDB Firestore",
    ]

    if _question_mentions_any(q, ["tech stack", "technology stack", "technologies", "tools", "stack"]):
        return _dedupe_preserve_order(base_queries + tech_stack_queries)[:GENERAL_CAREER_QUERY_LIMIT]

    if _question_mentions_any(q, ["cloud engineering", "cloud skills", "cloud experience", "cloud platform", "cloud platforms", "aws", "gcp", "azure"]):
        return _dedupe_preserve_order(base_queries + cloud_queries + ["Stephen Adegbile AWS and GCP production delivery"])[:GENERAL_CAREER_QUERY_LIMIT]

    if _question_mentions_any(q, ["data engineering", "etl", "elt", "pipelines", "data pipeline", "lakehouse", "warehouse", "warehousing", "spark", "airflow", "databricks", "snowflake", "bigquery"]):
        return _dedupe_preserve_order(base_queries + data_eng_queries + ["Stephen Adegbile AWS and GCP data pipelines analytics ready datasets"])[:GENERAL_CAREER_QUERY_LIMIT]

    if _question_mentions_any(q, ["ai engineering", "ai skills", "ai experience", "ml", "machine learning", "rag", "agent", "agents", "llm"]):
        return _dedupe_preserve_order(base_queries + ai_eng_queries)[:GENERAL_CAREER_QUERY_LIMIT]

    if _question_mentions_any(q, ["software engineering", "backend", "frontend", "full stack", "full-stack", "api", "apis", "web development"]):
        return _dedupe_preserve_order(base_queries + software_queries)[:GENERAL_CAREER_QUERY_LIMIT]

    return _dedupe_preserve_order(base_queries + tech_stack_queries[:4])[:GENERAL_CAREER_QUERY_LIMIT]


def _fallback_retrieval_queries(question: str, intent: str) -> list[str]:
    question = _normalize_whitespace(question)
    if intent != "career":
        return [question[:1200]]

    if not _is_job_fit_question(question) and len(question) < 1000:
        return _build_general_career_queries(question)

    queries = [
        "Stephen Adegbile skills experience role fit",
        "Stephen Adegbile production AI systems delivery",
        "Stephen Adegbile architecture project delivery experience",
    ]
    for key in _extract_requirement_keys(question):
        queries.append(_skill_query_from_key(key))
    return _dedupe_preserve_order(queries)[:JOB_FIT_QUERY_LIMIT]


def _plan_retrieval_queries(state: AgentState, llm) -> list[str]:
    question = state["question"]
    intent = state["intent"]

    if intent != "career":
        return _fallback_retrieval_queries(question, intent)

    if len(question) < 700 and not _is_job_fit_question(question):
        return _build_general_career_queries(question)

    try:
        planner = llm.with_structured_output(RetrievalPlan)
        prompt = RETRIEVAL_PLANNER_PROMPT.format_messages(
            conversation_history=_format_history(state.get("conversation_history", [])),
            question=question,
        )
        plan = planner.invoke(prompt)

        queries = [
            "Stephen Adegbile skills experience role fit",
            "Stephen Adegbile production AI systems delivery",
            "Stephen Adegbile architecture project delivery experience",
        ]

        if plan.condensed_focus:
            queries.append(plan.condensed_focus)

        queries.extend(plan.evidence_queries or [])

        for key in _extract_requirement_keys(question):
            queries.append(_skill_query_from_key(key))

        queries = _dedupe_preserve_order(queries)
        return queries[:JOB_FIT_QUERY_LIMIT] if queries else _fallback_retrieval_queries(question, intent)

    except Exception:
        logger.exception("Retrieval planning failed, using deterministic fallback.")
        return _fallback_retrieval_queries(question, intent)


def _embed_queries(client_genai, queries: list[str]) -> list[list[float]]:
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []

    embed_res = client_genai.models.embed_content(
        model=os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001"),
        contents=queries,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    embeddings = []
    for emb in embed_res.embeddings:
        vector = emb.values
        if len(vector) != 3072:
            raise ValueError(f"Embedding dimension mismatch: expected 3072, got {len(vector)}")
        embeddings.append(vector)
    return embeddings


def _aggregate_ranked_docs(results: dict, max_docs: int) -> list[str]:
    docs_nested = results.get("documents", []) or []
    distances_nested = results.get("distances", []) or []
    best_by_doc = {}

    for docs, distances in zip(docs_nested, distances_nested):
        for doc, dist in zip(docs or [], distances or []):
            if not doc:
                continue
            if best_by_doc.get(doc) is None or dist < best_by_doc[doc]:
                best_by_doc[doc] = dist

    return [doc for doc, _ in sorted(best_by_doc.items(), key=lambda x: x[1])[:max_docs]]


def _query_docs_for_queries(
    client_genai,
    collection,
    queries: list[str],
    per_query_results: int,
    max_docs: int,
) -> list[str]:
    queries = _dedupe_preserve_order(queries)
    if not queries:
        return []

    query_vectors = _embed_queries(client_genai, queries)
    if not query_vectors:
        return []

    results = collection.query(
        query_embeddings=query_vectors,
        n_results=per_query_results,
        include=["documents", "distances"],
    )
    return _aggregate_ranked_docs(results, max_docs)


def _merge_docs_preserve_order(*doc_lists: list[str], max_docs: int) -> list[str]:
    merged = []
    seen = set()

    for docs in doc_lists:
        for doc in docs:
            if doc and doc not in seen:
                seen.add(doc)
                merged.append(doc)
                if len(merged) >= max_docs:
                    return merged

    return merged


def _build_requirement_bucket_queries(question: str) -> dict[str, list[str]]:
    requirement_keys = _extract_requirement_keys(question)

    buckets: dict[str, list[str]] = {}
    for key in requirement_keys:
        buckets[key] = [_skill_query_from_key(key)]

    if "aws" in requirement_keys:
        buckets["aws"] = _dedupe_preserve_order([
            "Stephen Adegbile AWS experience cloud delivery architecture",
            "Stephen Adegbile AWS S3 Glue EMR MWAA Kinesis Lambda Redshift Athena SageMaker",
            "Stephen Adegbile AWS data pipelines ETL analytics ready datasets",
        ])

    if "gcp" in requirement_keys:
        buckets["gcp"] = _dedupe_preserve_order([
            "Stephen Adegbile GCP experience cloud delivery architecture",
            "Stephen Adegbile GCP Cloud Functions Gen2 GCS BigQuery Vertex AI Cloud Composer PubSub Cloud Run",
            "Stephen Adegbile GCP data pipelines analytics ready datasets",
        ])

    if "azure" in requirement_keys:
        buckets["azure"] = _dedupe_preserve_order([
            "Stephen Adegbile Azure experience",
            "Stephen Adegbile Azure App Service startup valuation deployment",
        ])

    if "security" in requirement_keys:
        buckets["security"] = _dedupe_preserve_order([
            "Stephen Adegbile security compliance governance delivery",
            "Stephen Adegbile Secret Manager API key security schema enforcement data quality governance",
        ])

    if "stakeholder" in requirement_keys:
        buckets["stakeholder"] = _dedupe_preserve_order([
            "Stephen Adegbile stakeholder client communication presentations",
            "Stephen Adegbile collaborated with stakeholders requirements dashboards decision making",
        ])

    if "airflow" in requirement_keys:
        buckets["airflow"] = _dedupe_preserve_order([
            "Stephen Adegbile Apache Airflow MWAA Cloud Composer orchestration delivery",
            "Stephen Adegbile Databricks Workflows Cloud Composer MWAA orchestration",
        ])

    if "databricks" in requirement_keys:
        buckets["databricks"] = _dedupe_preserve_order([
            "Stephen Adegbile Databricks Delta Lake Unity Catalog Delta Live Tables delivery",
            "Stephen Adegbile Databricks Workflows medallion architecture regulatory reporting",
        ])

    if "snowflake" in requirement_keys:
        buckets["snowflake"] = _dedupe_preserve_order([
            "Stephen Adegbile Snowflake Snowpipe dbt data warehouse delivery",
            "Stephen Adegbile fraud detection Snowflake Terraform dbt analytics",
        ])

    return buckets

# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------

def load_resources():
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES

    with _RESOURCE_LOCK:
        if _RESOURCES is not None:
            return _RESOURCES

        bucket_name = os.getenv("GCS_BUCKET")
        project_id = os.getenv("GCP_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "europe-west2")
        db_dir = Path("/tmp/chroma_db")
        tar_path = Path("/tmp/chroma_db.tar.gz")

        if not bucket_name:
            raise ValueError("Missing required environment variable: GCS_BUCKET")
        if bucket_name == "your-bucket-name":
            raise ValueError("GCS_BUCKET is still set to placeholder value 'your-bucket-name'")
        if not project_id:
            raise ValueError("Missing required environment variable: GCP_PROJECT")

        if not db_dir.exists():
            logger.info("Chroma DB not present in /tmp. Downloading from GCS...")
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob("chroma_db.tar.gz")

            if not blob.exists():
                raise FileNotFoundError(f"chroma_db.tar.gz not found in bucket {bucket_name}")

            blob.download_to_filename(str(tar_path))
            with tarfile.open(str(tar_path), "r:gz") as tar:
                tar.extractall("/tmp/")

        chroma_client = chromadb.PersistentClient(path=str(db_dir))
        collection = chroma_client.get_collection(name="portfolio")

        llm = ChatVertexAI(
            model=os.getenv("VERTEX_CHAT_MODEL", "gemini-2.5-flash"),
            project=project_id,
            location=location,
            temperature=0,
            max_retries=2,
        )

        client_genai = genai.Client(vertexai=True, project=project_id, location=location)

        _RESOURCES = {
            "collection": collection,
            "llm": llm,
            "client_genai": client_genai,
        }
        logger.info("Resources loaded successfully.")
        return _RESOURCES


def append_history(state: AgentState, role: str, content: str) -> list[ConversationTurn]:
    history = list(state.get("conversation_history", []))
    history.append({"role": role, "content": content})
    return history


def render_final_response(intent: str, structured: dict) -> str:
    if intent == "career":
        competencies = "\n".join(f"- {c}" for c in structured["core_competencies"])
        return (
            f"SUMMARY:\n{structured['summary']}\n\n"
            f"CORE COMPETENCIES:\n{competencies}\n\n"
            f"STRATEGIC VALUE:\n{structured['strategic_value']}"
        )
    if intent == "project":
        highlights = "\n".join(f"- {h}" for h in structured["architectural_highlights"])
        stack = "\n".join(f"- {t}" for t in structured["tech_stack_tooling"])
        missing = structured.get("missing_context")
        missing_block = f"\n\nMISSING CONTEXT:\n{missing}" if missing else ""
        return (
            f"## {structured['project_name']}\n\n"
            f"**Problem Statement:** {structured['problem_statement']}\n\n"
            f"**Architecture & Workflow:** {structured['architecture_workflow']}\n\n"
            f"**Tech Stack & Tooling:**\n{stack}\n\n"
            f"**Architectural Highlights:**\n{highlights}"
            f"{missing_block}"
        )
    if intent == "schedule":
        competencies = "\n".join(f"- {c}" for c in structured["competencies"])
        return (
            f"{structured['summary']}\n\n"
            f"COMPETENCIES / AGENDA:\n{competencies}\n\n"
            f"STRATEGIC VALUE:\n{structured['strategic_value']}\n\n"
            f"Book via Calendly: {structured['booking_link']}\n"
            f"Direct Email: {structured['direct_email']}"
        )
    return json.dumps(structured, indent=2)

# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------

def initialize_state(state: AgentState) -> AgentState:
    question = state["question"].strip()
    role = (state.get("role") or "general").lower()
    is_job_fit = _is_job_fit_question(question)

    history = [{"role": "user", "content": question}]
    return {
        "question": question,
        "role": role,
        "conversation_history": history,
        "is_job_fit": is_job_fit,
    }


def classify_route(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(RouteDecision)

    prompt = ROUTER_PROMPT.format_messages(
        conversation_history=_format_history(state.get("conversation_history", [])),
        question=state["question"],
        role=state.get("role", "general"),
    )
    route = llm.invoke(prompt)

    explicit_role = state.get("role", "general")
    user_type = explicit_role if explicit_role in {"recruiter", "engineer", "hiring_manager", "student"} else route.inferred_user_type
    return {"intent": route.intent, "user_type": user_type}


def retrieve_context(state: AgentState) -> AgentState:
    resources = load_resources()
    client_genai = resources["client_genai"]
    collection = resources["collection"]
    llm = resources["llm"]
    intent = state["intent"]

    if intent == "career":
        if state.get("is_job_fit", False):
            retrieval_queries = _plan_retrieval_queries(state, llm)
            logger.info("Career job-fit retrieval queries: %s", retrieval_queries)

            general_docs = _query_docs_for_queries(
                client_genai=client_genai,
                collection=collection,
                queries=retrieval_queries,
                per_query_results=CAREER_PER_QUERY_RESULTS,
                max_docs=JOB_FIT_GENERAL_DOCS,
            )

            bucket_queries = _build_requirement_bucket_queries(state["question"])
            logger.info("Career job-fit requirement buckets: %s", list(bucket_queries.keys()))

            bucket_docs_lists = []
            for bucket_name, bucket_query_list in bucket_queries.items():
                docs_for_bucket = _query_docs_for_queries(
                    client_genai=client_genai,
                    collection=collection,
                    queries=bucket_query_list,
                    per_query_results=CAREER_PER_QUERY_RESULTS,
                    max_docs=JOB_FIT_BUCKET_DOCS,
                )
                logger.info(
                    "Career job-fit bucket '%s' retrieved %s docs",
                    bucket_name,
                    len(docs_for_bucket),
                )
                bucket_docs_lists.append(docs_for_bucket)

            docs = _merge_docs_preserve_order(
                general_docs,
                *bucket_docs_lists,
                max_docs=MAX_RETRIEVAL_DOCS,
            )

            return {
                "retrieved_docs": docs,
                "retrieved_context": "\n\n".join(docs),
                "retrieval_queries": retrieval_queries,
            }

        retrieval_queries = _build_general_career_queries(state["question"])
        logger.info("Career general retrieval queries: %s", retrieval_queries)

        query_vectors = _embed_queries(client_genai, retrieval_queries)
        results = collection.query(
            query_embeddings=query_vectors,
            n_results=CAREER_PER_QUERY_RESULTS,
            include=["documents", "distances"],
        )
        docs = _aggregate_ranked_docs(results, MAX_RETRIEVAL_DOCS)
        return {
            "retrieved_docs": docs,
            "retrieved_context": "\n\n".join(docs),
            "retrieval_queries": retrieval_queries,
        }

    embed_res = client_genai.models.embed_content(
        model=os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001"),
        contents=[state["question"]],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = embed_res.embeddings[0].values
    logger.info("Query embedding length: %s", len(query_vector))

    if len(query_vector) != 3072:
        raise ValueError(f"Embedding dimension mismatch: expected 3072, got {len(query_vector)}")

    n_results = 4 if intent == "project" else 2 if intent == "schedule" else 6
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents"],
    )
    docs = results.get("documents", [[]])[0] if results else []
    return {
        "retrieved_docs": docs,
        "retrieved_context": "\n\n".join(docs),
        "retrieval_queries": [state["question"]],
    }


def generate_career_response(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(CareerResponse)

    prompt = CAREER_PROMPT.format_messages(
        user_type=state.get("user_type", "general"),
        conversation_history=_format_history(state.get("conversation_history", [])),
        context=state.get("retrieved_context", ""),
        question=state["question"],
        is_job_fit=state.get("is_job_fit", False),
    )
    result = llm.invoke(prompt)
    return {"structured_response": result.model_dump()}


def generate_project_response(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(ProjectResponse)

    prompt = PROJECT_PROMPT.format_messages(
        user_type=state.get("user_type", "general"),
        conversation_history=_format_history(state.get("conversation_history", [])),
        context=state.get("retrieved_context", ""),
        question=state["question"],
    )
    result = llm.invoke(prompt)
    return {"structured_response": result.model_dump()}


def generate_schedule_response(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(ScheduleResponse)

    prompt = SCHEDULE_PROMPT.format_messages(
        user_type=state.get("user_type", "general"),
        question=state["question"],
    )
    result = llm.invoke(prompt)
    return {"structured_response": result.model_dump()}


def refine_response(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(GroundingCheck)

    candidate_answer = render_final_response(state["intent"], state["structured_response"])
    prompt = REFINER_PROMPT.format_messages(
        is_job_fit=state.get("is_job_fit", False),
        context=state.get("retrieved_context", ""),
        candidate_answer=candidate_answer,
    )
    result = llm.invoke(prompt)

    final_answer = candidate_answer if result.grounded else (result.revised_answer or candidate_answer)
    history = append_history(state, "assistant", final_answer)
    return {
        "needs_refine": not result.grounded,
        "grounding_notes": result.notes,
        "refined_response": final_answer,
        "answer": final_answer,
        "conversation_history": history,
    }

# -----------------------------------------------------------------------------
# Routing helpers / graphs
# -----------------------------------------------------------------------------

def route_after_classifier(state: AgentState) -> Literal["career_graph", "project_graph", "schedule_graph"]:
    return {"career": "career_graph", "project": "project_graph", "schedule": "schedule_graph"}[state["intent"]]


def build_career_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_context", retrieve_context, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    builder.add_node("generate_career_response", generate_career_response, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    builder.add_node("refine_response", refine_response, retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0))
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_career_response")
    builder.add_edge("generate_career_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


def build_project_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_context", retrieve_context, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    builder.add_node("generate_project_response", generate_project_response, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    builder.add_node("refine_response", refine_response, retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0))
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_project_response")
    builder.add_edge("generate_project_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


def build_schedule_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_context", retrieve_context, retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0))
    builder.add_node("generate_schedule_response", generate_schedule_response, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    builder.add_node("refine_response", refine_response, retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0))
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_schedule_response")
    builder.add_edge("generate_schedule_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


def get_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    with _GRAPH_LOCK:
        if _GRAPH is not None:
            return _GRAPH

        career_graph = build_career_subgraph()
        project_graph = build_project_subgraph()
        schedule_graph = build_schedule_subgraph()

        builder = StateGraph(AgentState)
        builder.add_node("initialize_state", initialize_state)
        builder.add_node("classify_route", classify_route, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
        builder.add_node("career_graph", career_graph)
        builder.add_node("project_graph", project_graph)
        builder.add_node("schedule_graph", schedule_graph)

        builder.add_edge(START, "initialize_state")
        builder.add_edge("initialize_state", "classify_route")
        builder.add_conditional_edges("classify_route", route_after_classifier)
        builder.add_edge("career_graph", END)
        builder.add_edge("project_graph", END)
        builder.add_edge("schedule_graph", END)

        _GRAPH = builder.compile()
        return _GRAPH

# -----------------------------------------------------------------------------
# Cloud Function entry point
# -----------------------------------------------------------------------------

@functions_framework.http
def agent_router(request):
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    headers = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}

    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        role = (data.get("role") or "general").lower()
        thread_id = (data.get("thread_id") or data.get("session_id") or "default-session").strip()

        if not question:
            return (json.dumps({"error": "No question provided"}), 400, headers)

        graph = get_graph()
        initial_state: AgentState = {
            "question": question,
            "role": role,
            "thread_id": thread_id,
        }

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = graph.invoke(initial_state, config=config)

        payload = {
            "answer": result.get("answer", ""),
            "intent": result.get("intent", "career"),
            "role": result.get("user_type", role),
            "thread_id": thread_id,
            "grounding_notes": result.get("grounding_notes", ""),
            "needs_refine": result.get("needs_refine", False),
        }
        return (json.dumps(payload), 200, headers)

    except Exception as e:
        logger.exception("Runtime Error")
        return (json.dumps({"error": str(e)}), 500, headers)
