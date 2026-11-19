"""
Module: answer_generator.py
Description: Generates context-grounded answers to user queries using retrieved document
passages and Gemini LLM. Strictly instructs the model to cite sources and acknowledge missing evidence.
"""

from dataclasses import dataclass
from typing import List, Optional
from backend.generation.gemini_client import GeminiClient
from backend.retrieval.vector_store import ChromaVectorStore, RetrievalResult


@dataclass
class GeneratedAnswer:
    """Encapsulates the generated answer and the source context passages used."""
    question: str
    answer_text: str
    retrieved_passages: List[RetrievalResult]
    is_grounded_fallback: bool = False
    warning: Optional[str] = None


class AnswerGenerator:
    """Constructs grounded prompts and queries Gemini for document question answering."""

    SYSTEM_INSTRUCTION = (
        "You are GROUND LENS, an objective, rigorous AI technical assistant. "
        "Your task is to answer the user's question accurately and exclusively based on the provided context passages. "
        "Strict rules:\n"
        "1. Answer ONLY using the facts stated in the provided context passages.\n"
        "2. Do NOT invent, assume, or extrapolate numbers, dates, benchmarks, or technical metrics.\n"
        "3. If the context does not contain sufficient information to answer the question or part of the question, "
        "explicitly state: 'The provided document does not contain information regarding [topic]'.\n"
        "4. Be precise, concise, and technical."
    )

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        top_k: int = 4,
    ):
        self.client = gemini_client or GeminiClient()
        self.vector_store = vector_store or ChromaVectorStore()
        self.top_k = top_k

    def build_context_block(self, passages: List[RetrievalResult]) -> str:
        """Formats retrieved chunks with clear provenance metadata."""
        if not passages:
            return "No relevant context passages found in indexed documents."

        context_lines = []
        for idx, p in enumerate(passages, 1):
            context_lines.append(
                f"--- EVIDENCE PASSAGE {idx} (Document: {p.doc_name}, Page: {p.page_number}) ---\n"
                f"{p.text.strip()}\n"
            )
        return "\n".join(context_lines)

    def generate_answer(
        self,
        question: str,
        doc_name: Optional[str] = None,
        custom_top_k: Optional[int] = None,
    ) -> GeneratedAnswer:
        """
        Retrieves relevant passages for the question and generates a grounded response.
        """
        k = custom_top_k or self.top_k
        passages = self.vector_store.search(question, top_k=k, doc_name=doc_name)

        if not passages:
            return GeneratedAnswer(
                question=question,
                answer_text="No documents or relevant passages found matching your question. Please upload and index a PDF first.",
                retrieved_passages=[],
                warning="No context retrieved from vector store.",
            )

        context_block = self.build_context_block(passages)

        prompt = (
            f"CONTEXT PASSAGES FROM SOURCE DOCUMENTS:\n"
            f"{context_block}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"GROUNDED ANSWER:"
        )

        try:
            answer_text = self.client.generate_content(
                prompt=prompt,
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.2,
            )
            return GeneratedAnswer(
                question=question,
                answer_text=answer_text,
                retrieved_passages=passages,
            )
        except Exception as e:
            # If API key is missing or call fails, provide informative message
            return GeneratedAnswer(
                question=question,
                answer_text=f"[Answer Generation Notice]: Unable to query Gemini API ({e}). "
                            "Please ensure a valid GEMINI_API_KEY is configured in .env or the sidebar.",
                retrieved_passages=passages,
                warning=str(e),
                is_grounded_fallback=True,
            )
