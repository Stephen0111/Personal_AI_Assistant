# Personal AI Assistant (Multi-Agent RAG System)
**🔗 Live Demo:** [https://huggingface.co/spaces/Stephen0111/personal-ai-assistant](https://huggingface.co/spaces/Stephen0111/personal-ai-assistant)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![GCP](https://img.shields.io/badge/Cloud-GCP-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

---

## Overview

This project is a cloud-native **multi-agent, retrieval-augmented personal AI assistant** embedded into a portfolio website.

It is designed to act as a **24/7 intelligent representative**, capable of:
- Answering recruiter questions about my background and experience
- Explaining my projects with architecture-level technical depth
- Demonstrating real AI engineering, RAG, and orchestration capabilities
- Converting visitors into scheduled meetings

This system showcases **production-grade AI architecture**, combining:
- **LangGraph-based multi-agent orchestration**
- **Retrieval-Augmented Generation (RAG)**
- **Semantic intent routing**
- **Stateful conversation memory**
- **Grounded response refinement**
- **Cloud deployment on GCP**

---

## Features

### Smart Portfolio Assistant
- Streamlit-based chatbot UI
- Role-aware responses for:
  - Recruiters
  - Engineers
  - Hiring Managers
  - Students
  - General visitors
- Multi-turn memory using session/thread state
- Project explanations with architecture-first answers
- Scheduling flow for recruiter and hiring-manager conversations

---

### Multi-Agent Architecture
- **Router Agent** → LLM-based semantic intent classification
- **Career Expert Agent** → CV, experience, skills, and business-value answers
- **Project Architect Agent** → Technical deep-dives into systems, workflows, and stack decisions
- **Scheduling Coordinator Agent** → Meeting conversion via Calendly

---

### LangGraph Orchestration
- Built with **LangGraph StateGraph**
- Uses **conditional routing** instead of keyword-based if/else logic
- Supports **specialized subgraphs** for each intent
- Uses **RetryPolicy** for resilience instead of manual retry loops
- Compiled with a **checkpointer** for session-aware memory

---

### Stateful Memory & Session Context
- Tracks:
  - user role
  - inferred persona
  - conversation history
  - thread/session ID
- Uses **MemorySaver checkpointer**
- Supports multi-turn recruiter and engineering conversations
- Preserves context across turns within the same session

---

### Structured Output & Response Quality
- Prompting built with **ChatPromptTemplate**
- Output schemas enforced using **Pydantic**
- Returns structured responses for:
  - Career answers
  - Project deep-dives
  - Scheduling workflows
- Includes a **grounding / refiner step** that checks generated answers against retrieved ChromaDB context to reduce hallucinations

---

### Retrieval-Augmented Generation (RAG)
- Vector database: **ChromaDB**
- Embeddings: **Google Gemini Embeddings (`gemini-embedding-001`)**
- Retrieval context loaded from a persistent Chroma collection stored in **Google Cloud Storage**
- Supports semantic retrieval across:
  - CV content
  - Project documentation
  - GitHub project context
  - Portfolio engineering content

---

### Cloud-Native Backend
- Deployable as a **Google Cloud Function / Cloud Run Function**
- CORS-enabled HTTP entry point
- Singleton resource loading for:
  - ChromaDB
  - Vertex AI chat model
  - Google GenAI embeddings client
- ChromaDB downloaded from GCS to `/tmp` at runtime for stateless cloud execution

---

## Architecture Overview

![Architecture Overview](./assets/473A70AB-C566-419D-A709-51080B785B6D.png)

---

## System Design

### High-Level Flow

1. User interacts with the Streamlit UI
2. Request is sent to the Cloud Function backend
3. LangGraph initializes graph state for the current session
4. Router Agent semantically classifies the query
5. Request is routed to one of three specialist subgraphs:
   - Career Expert
   - Project Architect
   - Scheduling Coordinator
6. Relevant context is retrieved from ChromaDB
7. Specialist agent generates a structured response
8. Refiner node validates the response against retrieved context
9. Final grounded answer is returned to the frontend
10. Session memory is persisted via checkpointer

---

## Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| Backend | Python, Google Cloud Functions |
| Agent Orchestration | LangGraph |
| Prompting / LLM Framework | LangChain Core |
| LLM | Vertex AI Gemini 2.5 Flash |
| Embeddings | Google GenAI SDK, `gemini-embedding-001` |
| Vector DB | ChromaDB |
| Structured Output | Pydantic |
| State / Memory | LangGraph Checkpointer (`MemorySaver`) |
| Prompt Templates | `ChatPromptTemplate` |
| Cloud Storage | Google Cloud Storage |
| Cloud Platform | GCP |
| Container / Runtime | Cloud Functions Gen 2 / Cloud Run Functions |

---

## Multi-Agent Architecture

### Router Agent
- Uses **LLM-based semantic classification**
- Detects whether a user query is best handled as:
  - `career`
  - `project`
  - `schedule`
- Infers likely user persona where possible

### Career Expert Agent
- Answers questions about:
  - Background
  - Experience
  - Skills
  - Career trajectory
  - Business value
- Tailors technical depth based on user role

### Project Architect Agent
- Explains:
  - Problem statement
  - Architecture and workflow
  - Stack and tooling
  - Engineering trade-offs
  - Architectural highlights
- Prioritizes system design over generic summaries

### Scheduling Coordinator Agent
- Handles:
  - Booking requests
  - Intro call intent
  - Recruiter / HM follow-up
- Directs users to:
  - Calendly
  - direct email

### Refiner / Grounding Agent
- Compares the generated draft with retrieved RAG context
- Rejects unsupported claims
- Rewrites answers to stay grounded in verified context

---

## LangGraph State Schema

The assistant maintains a graph state that includes:
- `question`
- `role`
- `user_type`
- `thread_id`
- `conversation_history`
- `intent`
- `retrieved_context`
- `retrieved_docs`
- `structured_response`
- `grounding_notes`
- `needs_refine`
- `answer`

This enables **session-aware, role-aware, and retrieval-grounded multi-turn conversations**.

---

## RAG Pipeline

### Data Sources
- CV text
- Project summaries
- Portfolio content
- GitHub-related project documentation
- Other curated career/project context documents

### Pipeline Steps
1. Source document ingestion
2. Chunking and preparation
3. Embedding generation with `gemini-embedding-001`
4. Storage in ChromaDB
5. Chroma archive persisted to GCS
6. Semantic retrieval at query time
7. Grounded generation via specialist agents

---

## ChromaDB / GCS Runtime Loading

### Runtime Strategy
- ChromaDB archive is stored as `chroma_db.tar.gz` in GCS
- On cold start:
  - archive is downloaded to `/tmp`
  - extracted locally
  - mounted via `chromadb.PersistentClient`
- Resources are cached using a singleton-style loader per warm instance

### Why This Approach?
- Keeps the architecture **serverless and stateless**
- Avoids rebuilding the vector store on every request
- Makes the function compatible with Cloud Functions / Cloud Run runtime behavior
- Demonstrates production-minded cloud execution patterns

---

## Prompt Engineering & Output Control

### Prompt Templates
The system uses **ChatPromptTemplate** prompts for:
- Routing
- Career responses
- Project responses
- Scheduling responses
- Grounding / refinement

### Structured Outputs
Responses are validated with **Pydantic schemas**, including:
- `RouteDecision`
- `CareerResponse`
- `ProjectResponse`
- `ScheduleResponse`
- `GroundingCheck`

### Why This Matters
- Improves response consistency
- Makes outputs easier to render and debug
- Reduces formatting drift
- Improves reliability for recruiter-facing production usage

---

## Resilience & Reliability

### Built-In Reliability Mechanisms
- **LangGraph RetryPolicy** on key nodes
- Singleton resource loading to reduce repeated startup cost
- Explicit environment variable validation
- Embedding dimension checks to prevent Chroma mismatch errors
- Structured refinement stage for response grounding

### Design Benefits
- More resilient than manual retry loops
- Cleaner orchestration logic
- Better separation of concerns
- Easier extension for future agents/tools

---

## Scheduling Workflow

1. User asks to book or connect
2. Router classifies intent as `schedule`
3. Scheduling Coordinator generates a concise meeting response
4. User is directed to:
   - **Calendly:** https://calendly.com/stephen-adegbile19/new-meeting
   - **Email:** stephen@stephenadegbile.uk
5. Response is still passed through the grounding/refiner stage for consistency

---

## Deployment Architecture

### Backend
- Python HTTP function using `functions_framework`
- Deployable to:
  - **Google Cloud Functions Gen 2**
  - **Cloud Run Functions**

### Runtime Dependencies
- Vertex AI Gemini
- Google GenAI SDK
- ChromaDB
- LangGraph
- LangChain Core
- Pydantic
- Google Cloud Storage client

### Environment Variables
- `GCP_PROJECT`
- `GCS_BUCKET`
- `VERTEX_LOCATION`
- `VERTEX_CHAT_MODEL`
- `VERTEX_EMBED_MODEL`

---

## Local Development

### Prerequisites
- Python 3.10+
- pip / virtualenv
- Git
- GCP project access
- Vertex AI enabled
- ChromaDB archive stored in GCS

### Setup

```bash
git clone https://github.com/Stephen0111/personal-ai-assistant.git
cd personal-ai-assistant

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
