import asyncio
import logging
from app.config import get_settings
from app.vectorstore.chroma_client import get_vector_store
from app.models.schemas import SubQuery, RetrievedDocument

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrieverAgent:
    """Executes parallel vector searches for all sub-queries against ChromaDB."""

    def __init__(self):
        self.store = get_vector_store()

    async def retrieve(
        self,
        sub_queries: list[SubQuery],
        top_k: int | None = None,
        collection: str | None = None,
    ) -> list[RetrievedDocument]:
        """Run all sub-query searches concurrently and deduplicate results."""
        k = top_k or settings.top_k_results

        tasks = [
            asyncio.to_thread(
                self.store.similarity_search,
                sq.text,
                k=k,
                collection=collection,
                score_threshold=settings.similarity_threshold,
            )
            for sq in sub_queries
        ]

        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        seen_contents: set[str] = set()
        docs: list[RetrievedDocument] = []

        for result in results_per_query:
            if isinstance(result, Exception):
                logger.warning("Retrieval sub-query failed: %s", result)
                continue
            for item in result:
                fingerprint = item["content"][:120]
                if fingerprint not in seen_contents:
                    seen_contents.add(fingerprint)
                    docs.append(RetrievedDocument(**item))

        docs.sort(key=lambda d: d.score, reverse=True)
        logger.info("Retrieved %d unique documents across %d sub-queries", len(docs), len(sub_queries))
        return docs[:k]
