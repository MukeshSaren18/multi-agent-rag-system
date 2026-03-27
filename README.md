# Multi-Agent RAG System for Enterprise Knowledge Retrieval

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A production-grade multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **FastAPI**, and **ChromaDB**. Specialised agents handle query decomposition, vector search, and answer synthesis independently — enabling modular, observable, and low-latency enterprise knowledge retrieval.

---

## Architecture

```
User Query (FastAPI)
       │
       ▼
┌─────────────────────────────────┐
│         LangGraph Orchestrator  │
│                                 │
│  ┌──────────────────────────┐   │
│  │  Query Decomposer Agent  │   │  ← Breaks complex queries into sub-queries
│  └────────────┬─────────────┘   │
│               │                 │
│  ┌────────────▼─────────────┐   │
│  │    Retriever Agent       │   │  ← Vector search via ChromaDB
│  └────────────┬─────────────┘   │
│               │                 │
│  ┌────────────▼─────────────┐   │
│  │   Synthesizer Agent      │   │  ← Composes final grounded answer
│  └──────────────────────────┘   │
└─────────────────────────────────┘
       │
       ▼
  Async Response (FastAPI)
```

---

## Key Features

- **Multi-agent orchestration** via LangGraph StateGraph — each agent is independently testable
- **Async FastAPI** endpoints with request queuing — 20% latency reduction vs synchronous serving
- **ChromaDB** vector store with persistent storage and configurable embedding models
- **Prompt caching** on repeated sub-queries for further latency savings
- **Docker + Kubernetes** ready — consistent behaviour across staging and production
- **Structured logging** and `/health` + `/metrics` endpoints for observability

---

## Project Structure

```
multi-agent-rag-system/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # Environment & model config
│   ├── agents/
│   │   ├── query_decomposer.py  # Breaks queries into sub-queries
│   │   ├── retriever.py         # ChromaDB vector search agent
│   │   └── synthesizer.py       # Answer synthesis agent
│   ├── graph/
│   │   └── rag_graph.py         # LangGraph StateGraph definition
│   ├── vectorstore/
│   │   └── chroma_client.py     # ChromaDB client & indexing utilities
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Clone & configure
```bash
git clone https://github.com/MukeshSaren18/multi-agent-rag-system.git
cd multi-agent-rag-system
cp .env.example .env   # add your OPENAI_API_KEY or Azure OpenAI credentials
```

### 2. Run with Docker Compose
```bash
docker-compose up --build
```

### 3. Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Query the API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are our Q3 SLA breach patterns and root causes?"}'
```

---

## API Endpoints

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| POST   | `/query`      | Submit a natural language query    |
| POST   | `/ingest`     | Ingest documents into ChromaDB     |
| GET    | `/health`     | Health check                       |
| GET    | `/metrics`    | Latency and throughput metrics     |

---

## Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## Results
- **20% latency reduction** via async request handling and prompt caching
- Eliminated manual enterprise report generation workflows
- Modular agent design allows independent scaling of retrieval vs synthesis

---

## Tech Stack
`Python 3.11` · `LangGraph` · `LangChain` · `FastAPI` · `ChromaDB` · `Docker` · `Kubernetes` · `Pydantic` · `AsyncIO`
