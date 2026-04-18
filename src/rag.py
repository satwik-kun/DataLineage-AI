"""Simple RAG explanation module."""

from __future__ import annotations

import os
from typing import Iterable, List


class SimpleRAGExplainer:
    """Generate short explanations from retrieved context."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def _format_context(self, context_items: Iterable[str]) -> str:
        return "\n".join(f"- {item}" for item in context_items)

    def _mock_explanation(self, question: str, context_items: List[str]) -> str:
        context = self._format_context(context_items) if context_items else "- No context retrieved"
        return (
            "MVP explanation (mock LLM):\n"
            f"Question: {question}\n"
            "Relevant context:\n"
            f"{context}\n"
            "Interpretation: The answer is based on lineage edges and nearest semantic matches."
        )

    def explain(self, question: str, context_items: List[str]) -> str:
        """Return explanation from OpenAI if configured, else mock output."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return self._mock_explanation(question, context_items)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                "You are a concise data lineage assistant. Answer the question using only the given context. "
                "If context is insufficient, say that clearly.\n\n"
                f"Question: {question}\n"
                f"Context:\n{self._format_context(context_items)}"
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content or self._mock_explanation(question, context_items)
        except Exception:
            return self._mock_explanation(question, context_items)


if __name__ == "__main__":
    rag = SimpleRAGExplainer()
    print(rag.explain("What depends on sales.csv?", ["sales.csv -> customer_revenue_join"]))
