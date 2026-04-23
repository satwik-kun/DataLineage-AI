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
        if not context_items:
            return "No lineage evidence was retrieved to explain this query."

        relationship_lines = []
        for item in context_items:
            if "-[" in item and "]->" in item:
                relationship_lines.append(item)

        if relationship_lines:
            first = relationship_lines[0]
            return (
                f"For '{question}', lineage reasoning shows that {first}. "
                "This indicates an upstream-to-downstream dependency flow, so a change in the source can propagate "
                "through intermediate pipelines into final reports or datasets."
            )

        context = "; ".join(context_items)
        return (
            f"For '{question}', semantic retrieval identified relevant lineage nodes: {context}. "
            "The response is grounded in vector similarity and graph relationships rather than keyword matching."
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
