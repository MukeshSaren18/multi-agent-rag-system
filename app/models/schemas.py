from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Natural language query")
    collection: Optional[str] = Field(None, description="Override default ChromaDB collection")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")


class SubQuery(BaseModel):
    text: str
    intent: str


class RetrievedDocument(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedDocument]
    sub_queries: list[SubQuery]
    latency_ms: float
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1)
    metadatas: Optional[list[dict]] = None
    collection: Optional[str] = None


class IngestResponse(BaseModel):
    ingested_count: int
    collection: str
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    llm: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    cache_hit_rate: float
    error_rate: float
