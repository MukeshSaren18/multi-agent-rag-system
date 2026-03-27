import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.config import get_settings
from app.models.schemas import SubQuery

logger = logging.getLogger(__name__)
settings = get_settings()

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition specialist. Break the user query into focused 
sub-queries that can each be answered by retrieving specific documents. 

Return a JSON array of objects with keys:
- "text": the sub-query string
- "intent": one of [factual, analytical, comparative, procedural]

Decompose into 1-4 sub-queries. Return ONLY valid JSON, no markdown."""),
    ("human", "Query: {query}"),
])


class QueryDecomposerAgent:
    """Decomposes complex user queries into focused retrieval sub-queries."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        self.chain = DECOMPOSE_PROMPT | self.llm | JsonOutputParser()

    async def decompose(self, query: str) -> list[SubQuery]:
        """Break a query into sub-queries for parallel retrieval."""
        try:
            raw = await self.chain.ainvoke({"query": query})
            sub_queries = [SubQuery(text=item["text"], intent=item["intent"]) for item in raw]
            logger.info("Decomposed '%s' into %d sub-queries", query[:60], len(sub_queries))
            return sub_queries
        except Exception as e:
            logger.warning("Decomposition failed (%s), falling back to original query", e)
            return [SubQuery(text=query, intent="factual")]
