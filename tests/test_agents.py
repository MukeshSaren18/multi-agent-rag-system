"""
Unit tests for Multi-Agent RAG System
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.schemas import SubQuery, RetrievedDocument


# Mock ChatOpenAI for all tests to avoid API key validation errors
@pytest.fixture(autouse=True)
def mock_openai():
    """Mock ChatOpenAI before any agent imports"""
    with patch("langchain_openai.chat_models.ChatOpenAI"):
        yield


# ── Query Decomposer ────────────────────────────────────────────────────────

class TestQueryDecomposerAgent:
    @pytest.mark.asyncio
    async def test_decompose_returns_subqueries(self):
        from app.agents.query_decomposer import QueryDecomposerAgent

        agent = QueryDecomposerAgent()
        mock_output = [
            {"text": "What are Q3 SLA breaches?", "intent": "factual"},
            {"text": "Root causes of SLA failures?", "intent": "analytical"},
        ]

        with patch.object(agent, "chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_output)
            result = await agent.decompose("What are Q3 SLA breach patterns and root causes?")

        assert len(result) == 2
        assert all(isinstance(sq, SubQuery) for sq in result)
        assert result[0].intent == "factual"
        assert result[1].intent == "analytical"

    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_error(self):
        from app.agents.query_decomposer import QueryDecomposerAgent

        agent = QueryDecomposerAgent()
        with patch.object(agent, "chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
            result = await agent.decompose("test query")

        assert len(result) == 1
        assert result[0].text == "test query"
        assert result[0].intent == "factual"


# ── Retriever ───────────────────────────────────────────────────────────

class TestRetrieverAgent:
    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_results(self):
        from app.agents.retriever import RetrieverAgent

        agent = RetrieverAgent()
        duplicate_doc = {
            "content": "This is a duplicate document content that appears twice",
            "source": "doc1.pdf",
            "score": 0.92,
            "metadata": {},
        }

        with patch.object(agent.store, "similarity_search", return_value=[duplicate_doc]):
            sub_queries = [
                SubQuery(text="query A", intent="factual"),
                SubQuery(text="query B", intent="analytical"),
            ]
            results = await agent.retrieve(sub_queries)

        # Should deduplicate — same content fingerprint appears only once
        assert len(results) == 1
        assert results[0].score == 0.92

    @pytest.mark.asyncio
    async def test_retrieve_handles_failed_subquery(self):
        from app.agents.retriever import RetrieverAgent

        agent = RetrieverAgent()
        with patch.object(agent.store, "similarity_search", side_effect=Exception("DB error")):
            sub_queries = [SubQuery(text="failing query", intent="factual")]
            results = await agent.retrieve(sub_queries)

        assert results == []


# ── Synthesizer ──────────────────────────────────────────────────────────

class TestSynthesizerAgent:
    @pytest.mark.asyncio
    async def test_synthesize_returns_answer(self):
        from app.agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent()
        docs = [
            RetrievedDocument(content="Q3 had 12 SLA breaches.", source="report.pdf", score=0.91),
        ]

        mock_response = MagicMock()
        mock_response.content = "In Q3, there were 12 SLA breaches according to [Source: report.pdf]."

        with patch.object(agent, "chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_response)
            answer = await agent.synthesize("How many SLA breaches in Q3?", docs)

        assert "12 SLA breaches" in answer

    @pytest.mark.asyncio
    async def test_synthesize_no_documents(self):
        from app.agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent()
        answer = await agent.synthesize("anything", [])
        assert "No relevant documents" in answer


# ── FastAPI endpoints ────────────────────────────────────────────────────────

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        with patch("app.main.get_vector_store") as mock_store:
            mock_store.return_value.health_check = MagicMock(return_value=True)
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ("ok", "degraded")

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "avg_latency_ms" in data
        assert "cache_hit_rate" in data
