import logging
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChromaVectorStore:
    """Manages ChromaDB connection, indexing, and retrieval."""

    def __init__(self):
        self._client: Optional[chromadb.HttpClient] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._stores: dict[str, Chroma] = {}

    def connect(self):
        """Initialise ChromaDB client and embedding model."""
        try:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("ChromaDB connected at %s:%s", settings.chroma_host, settings.chroma_port)
        except Exception as e:
            logger.error("ChromaDB connection failed: %s", e)
            raise

    def get_store(self, collection: Optional[str] = None) -> Chroma:
        """Return (or create) a LangChain Chroma wrapper for a collection."""
        col = collection or settings.chroma_collection
        if col not in self._stores:
            self._stores[col] = Chroma(
                client=self._client,
                collection_name=col,
                embedding_function=self._embeddings,
            )
            logger.info("Initialised Chroma store for collection: %s", col)
        return self._stores[col]

    def ingest_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        collection: Optional[str] = None,
    ) -> int:
        """Embed and store documents. Returns number ingested."""
        store = self.get_store(collection)
        metas = metadatas or [{} for _ in texts]
        store.add_texts(texts=texts, metadatas=metas)
        logger.info("Ingested %d documents into '%s'", len(texts), collection or settings.chroma_collection)
        return len(texts)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        collection: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """Return top-k documents with similarity scores."""
        store = self.get_store(collection)
        results = store.similarity_search_with_relevance_scores(query, k=k)
        docs = []
        for doc, score in results:
            if score >= score_threshold:
                docs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": round(score, 4),
                    "metadata": doc.metadata,
                })
        return docs

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False


# Singleton
_vector_store = ChromaVectorStore()


def get_vector_store() -> ChromaVectorStore:
    return _vector_store
