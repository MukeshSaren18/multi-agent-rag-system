import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import get_settings
from app.models.schemas import RetrievedDocument

logger = logging.getLogger(__name__)
settings = get_settings()

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an enterprise knowledge assistant. Answer the user's question 
using ONLY the provided context documents. 

Rules:
- Ground every claim in the provided documents
- If the answer cannot be found in context, say so clearly
- Be concise and structured
- Cite sources where relevant using [Source: <source>] notation"""),
    ("human", """Question: {query}

Context Documents:
{context}

Answer:"""),
])


class SynthesizerAgent:
    """Composes a grounded, cited answer from retrieved documents."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
        )
        self.chain = SYNTHESIS_PROMPT | self.llm

    async def synthesize(self, query: str, documents: list[RetrievedDocument]) -> str:
        """Generate a grounded answer from retrieved context."""
        if not documents:
            return "No relevant documents found in the knowledge base for this query."

        context_blocks = []
        for i, doc in enumerate(documents, 1):
            context_blocks.append(
                f"[{i}] Source: {doc.source} (score: {doc.score:.2f})\n{doc.content}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        response = await self.chain.ainvoke({"query": query, "context": context})
        answer = response.content
        logger.info("Synthesized answer (%d chars) from %d documents", len(answer), len(documents))
        return answer
