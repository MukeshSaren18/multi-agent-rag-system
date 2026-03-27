import time
import logging
import asyncio
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.graph.rag_graph import rag_pipeline
from app.vectorstore.chroma_client import get_vector_store
from app.models.schemas import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, MetricsResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

# ── Simple in-memory metrics ──────────────────────────────────────────────────
_start_time = time.time()
_latencies: deque = deque(maxlen=1000)
_total_queries = 0
_errors = 0
_cache_hits = 0
_query_cache: dict[str, QueryResponse] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to ChromaDB…")
    get_vector_store().connect()
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent RAG System",
    description="Enterprise knowledge retrieval via LangGraph multi-agent pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query(request: QueryRequest):
    """Submit a natural language query and receive a grounded answer with sources."""
    global _total_queries, _errors, _cache_hits

    _total_queries += 1
    cache_key = f"{request.query}::{request.collection}::{request.top_k}"

    if settings.enable_prompt_cache and cache_key in _query_cache:
        _cache_hits += 1
        cached = _query_cache[cache_key]
        return QueryResponse(**{**cached.dict(), "cached": True})

    t0 = time.perf_counter()
    try:
        result = await rag_pipeline.ainvoke({
            "query": request.query,
            "collection": request.collection,
            "top_k": request.top_k,
            "sub_queries": [],
            "documents": [],
            "answer": "",
            "error": None,
        })
    except Exception as e:
        _errors += 1
        logger.error("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.perf_counter() - t0) * 1000
    _latencies.append(latency)

    response = QueryResponse(
        query=request.query,
        answer=result["answer"],
        sources=result["documents"],
        sub_queries=result["sub_queries"],
        latency_ms=round(latency, 2),
        cached=False,
    )

    if settings.enable_prompt_cache:
        _query_cache[cache_key] = response

    return response


@app.post("/ingest", response_model=IngestResponse, tags=["Indexing"])
async def ingest(request: IngestRequest):
    """Ingest raw text documents into ChromaDB for retrieval."""
    store = get_vector_store()
    count = await asyncio.to_thread(
        store.ingest_documents,
        request.documents,
        request.metadatas,
        request.collection,
    )
    col = request.collection or settings.chroma_collection
    return IngestResponse(ingested_count=count, collection=col)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """Liveness and dependency health check."""
    chroma_ok = await asyncio.to_thread(get_vector_store().health_check)
    return HealthResponse(
        status="ok" if chroma_ok else "degraded",
        chromadb="ok" if chroma_ok else "unreachable",
        llm="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Ops"])
async def metrics():
    """Runtime performance metrics."""
    avg = sum(_latencies) / len(_latencies) if _latencies else 0.0
    cache_rate = _cache_hits / _total_queries if _total_queries else 0.0
    error_rate = _errors / _total_queries if _total_queries else 0.0
    return MetricsResponse(
        total_queries=_total_queries,
        avg_latency_ms=round(avg, 2),
        cache_hit_rate=round(cache_rate, 4),
        error_rate=round(error_rate, 4),
    )
