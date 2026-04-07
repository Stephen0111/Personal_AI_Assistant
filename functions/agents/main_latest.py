import functions_framework
import json
import logging
import os
import tarfile
import threading
from pathlib import Path
from typing import Literal, Optional, TypedDict

from google.cloud import storage
from google import genai
from google.genai.types import EmbedContentConfig

import chromadb
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
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
_CHECKPOINTER = None
_GRAPH = None
_GRAPH_LOCK = threading.Lock()

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

    # Memory / state
    conversation_history: list[ConversationTurn]

    # Routing
    intent: Literal["career", "project", "schedule"]

    # Retrieval
    retrieved_context: str
    retrieved_docs: list[str]

    # Drafting / quality
    structured_response: dict
    refined_response: str
    grounding_notes: str
    needs_refine: bool

    # Final
    answer: str


# -----------------------------------------------------------------------------
# Pydantic schemas for structured outputs
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
    strategic_value: str = Field(description="How Stephen creates business value.")


class ProjectResponse(BaseModel):
    project_name: str = Field(description="Name of the project being discussed.")
    problem_statement: str = Field(description="Pain vs solution summary.")
    architecture_workflow: str = Field(description="High-level system/data flow.")
    tech_stack_tooling: list[str] = Field(description="Relevant technologies grouped or listed.")
    architectural_highlights: list[str] = Field(description="Key engineering wins.")
    missing_context: Optional[str] = Field(
        default=None,
        description="Any important missing context that should be stated explicitly."
    )


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
        description="If not grounded, return a corrected answer that uses only the context."
    )
    notes: str = Field(description="Brief explanation of what was changed or verified.")


# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a semantic router for Stephen Adegbile's portfolio assistant.
Choose the best destination:
- career: background, skills, experience, fit, CV, strengths
- project: project deep-dives, architecture, implementation, stack, system design
- schedule: booking, meeting, intro call, availability, next steps, confirmations to arrange time

Infer the user's role if possible:
recruiter, engineer, hiring_manager, student, or general.

Return structured output only."""
        ),
        (
            "human",
            """Conversation history:
{conversation_history}

Latest user question:
{question}

Current stated role from request:
{role}
"""
        ),
    ]
)

CAREER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Role: Senior Talent Solutions Agent for Stephen Adegbile.

Operating constraints:
- Use ONLY the provided context.
- If context is missing, say that the current portfolio context does not contain verified details.
- No hallucination.
- Adapt depth by user type:
  recruiter -> emphasize time-to-value, outcomes, production readiness
  engineer -> emphasize stack depth, architecture, scalability
  hiring_manager -> emphasize ownership, delivery, leadership signals
  student -> explain clearly without dumbing down

You must produce:
1) summary
2) core_competencies
3) strategic_value
"""
        ),
        (
            "human",
            """User type: {user_type}
Conversation history:
{conversation_history}

Retrieved context:
{context}

Question:
{question}
"""
        ),
    ]
)

PROJECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Role: Principal Solutions Architect / Technical Lead.

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
architectural_highlights, missing_context
"""
        ),
        (
            "human",
            """User type: {user_type}
Conversation history:
{conversation_history}

Retrieved context:
{context}

Question:
{question}
"""
        ),
    ]
)

SCHEDULE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Role: Executive Scheduling Coordinator.

Rules:
- Keep concise and action-oriented.
- Include booking link and direct email.
- Adapt meeting framing by user type:
  recruiter -> 15-min introductory sync
  engineer -> technical walkthrough
  hiring_manager -> strategic fit / delivery discussion
  general -> intro conversation
- Use structured output only.
"""
        ),
        (
            "human",
            """User type: {user_type}
Question:
{question}

Booking link: https://calendly.com/stephen-adegbile19/new-meeting
Direct email: stephen@stephenadegbile.uk
"""
        ),
    ]
)

REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grounding and hallucination checker.

Your job:
- Compare the candidate answer against the retrieved context.
- If unsupported claims appear, mark grounded=false and rewrite the answer using only supported facts.
- If the answer is supported, keep it.
- Be strict."""
        ),
        (
            "human",
            """Retrieved context:
{context}

Candidate answer:
{candidate_answer}
"""
        ),
    ]
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _format_history(history: list[ConversationTurn]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-8:])


def build_checkpointer():
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


def load_resources():
    """
    Singleton-style loader for Chroma + Vertex integrations.
    Downloads/extracts Chroma once per warm instance.
    """
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

        # Use the Google GenAI SDK for embeddings so retrieval matches the
        # original collection built with gemini-embedding-001 (3072 dims).
        client_genai = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

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
    """
    Ensures required defaults exist and appends the latest user question to memory.
    """
    question = state["question"].strip()
    role = (state.get("role") or "general").lower()
    history = append_history(
        {"conversation_history": state.get("conversation_history", [])},
        "user",
        question,
    )
    return {
        "question": question,
        "role": role,
        "conversation_history": history,
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
    user_type = explicit_role if explicit_role in {
        "recruiter", "engineer", "hiring_manager", "student"
    } else route.inferred_user_type

    return {
        "intent": route.intent,
        "user_type": user_type,
    }


def retrieve_context(state: AgentState) -> AgentState:
    resources = load_resources()
    client_genai = resources["client_genai"]
    collection = resources["collection"]

    embed_res = client_genai.models.embed_content(
        model=os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001"),
        contents=[state["question"]],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    query_vector = embed_res.embeddings[0].values
    logger.info("Query embedding length: %s", len(query_vector))

    if len(query_vector) != 3072:
        raise ValueError(
            f"Embedding dimension mismatch: expected 3072, got {len(query_vector)}"
        )

    intent = state["intent"]
    n_results = 4 if intent == "project" else 2 if intent == "schedule" else 6

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents"],
    )

    docs = results.get("documents", [[]])[0] if results else []
    context = "\n\n".join(docs)

    return {
        "retrieved_docs": docs,
        "retrieved_context": context,
    }


def generate_career_response(state: AgentState) -> AgentState:
    resources = load_resources()
    llm = resources["llm"].with_structured_output(CareerResponse)

    prompt = CAREER_PROMPT.format_messages(
        user_type=state.get("user_type", "general"),
        conversation_history=_format_history(state.get("conversation_history", [])),
        context=state.get("retrieved_context", ""),
        question=state["question"],
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

    candidate_answer = render_final_response(
        state["intent"],
        state["structured_response"],
    )

    prompt = REFINER_PROMPT.format_messages(
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
# Routing helpers
# -----------------------------------------------------------------------------

def route_after_classifier(state: AgentState) -> Literal["career_graph", "project_graph", "schedule_graph"]:
    return {
        "career": "career_graph",
        "project": "project_graph",
        "schedule": "schedule_graph",
    }[state["intent"]]


# -----------------------------------------------------------------------------
# Subgraphs
# -----------------------------------------------------------------------------

def build_career_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node(
        "retrieve_context",
        retrieve_context,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "generate_career_response",
        generate_career_response,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "refine_response",
        refine_response,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0),
    )

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_career_response")
    builder.add_edge("generate_career_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


def build_project_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node(
        "retrieve_context",
        retrieve_context,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "generate_project_response",
        generate_project_response,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "refine_response",
        refine_response,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0),
    )

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_project_response")
    builder.add_edge("generate_project_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


def build_schedule_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node(
        "retrieve_context",
        retrieve_context,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0),
    )
    builder.add_node(
        "generate_schedule_response",
        generate_schedule_response,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "refine_response",
        refine_response,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0),
    )

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "generate_schedule_response")
    builder.add_edge("generate_schedule_response", "refine_response")
    builder.add_edge("refine_response", END)
    return builder.compile()


# -----------------------------------------------------------------------------
# Parent graph
# -----------------------------------------------------------------------------

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
        builder.add_node(
            "classify_route",
            classify_route,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
        )

        builder.add_node("career_graph", career_graph)
        builder.add_node("project_graph", project_graph)
        builder.add_node("schedule_graph", schedule_graph)

        builder.add_edge(START, "initialize_state")
        builder.add_edge("initialize_state", "classify_route")
        builder.add_conditional_edges("classify_route", route_after_classifier)
        builder.add_edge("career_graph", END)
        builder.add_edge("project_graph", END)
        builder.add_edge("schedule_graph", END)

        _GRAPH = builder.compile(checkpointer=build_checkpointer())
        return _GRAPH


# -----------------------------------------------------------------------------
# Cloud Function entry point
# -----------------------------------------------------------------------------

@functions_framework.http
def agent_router(request):
    if request.method == "OPTIONS":
        return (
            "",
            204,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
    }

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

        config = {"configurable": {"thread_id": thread_id}}
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
