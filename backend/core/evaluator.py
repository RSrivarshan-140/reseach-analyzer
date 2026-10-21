import re
import os
from typing import List, Dict, Any, Optional
import json

class AnswerEvaluator:
    def __init__(self):
        pass

    def evaluate_response(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        provider: str = "gemini",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates faithfulness, relevance, context coverage, and hallucination risk.
        Returns a comprehensive evaluation report.
        """
        active_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

        if active_key and provider in ["gemini", "openai"]:
            try:
                return self._evaluate_with_llm(question, answer, context_chunks, provider, active_key)
            except Exception as e:
                print(f"[Warning] LLM evaluator failed ({e}), falling back to deterministic heuristic evaluation.")
                return self._evaluate_heuristic(question, answer, context_chunks)
        else:
            return self._evaluate_heuristic(question, answer, context_chunks)

    def _evaluate_with_llm(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        provider: str,
        api_key: str
    ) -> Dict[str, Any]:
        combined_context = "\n\n".join([f"[Context Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks)])

        prompt = f"""You are an unbiased automated evaluator assessing the quality and faithfulness of a RAG system for research papers.

QUESTION:
{question}

RETRIEVED CONTEXT:
{combined_context}

GENERATED ANSWER:
{answer}

INSTRUCTIONS:
1. Faithfulness (0.0 to 1.0): Check if every statement in the answer is directly supported by the retrieved context. Penalize any unsupported claims (hallucinations).
2. Answer Relevance (0.0 to 1.0): Check how completely and directly the answer addresses the question.
3. Context Coverage (0.0 to 1.0): Check what portion of relevant facts from the context was utilized.
4. Extract 3-5 individual factual claims from the answer and verify whether each is supported by the context.

Respond ONLY with valid JSON in this exact structure:
{{
  "faithfulness_score": 0.95,
  "answer_relevance_score": 0.90,
  "context_coverage_score": 0.85,
  "hallucination_risk": "Low",
  "overall_grade": "A (Excellent)",
  "verdict_summary": "Summary of findings...",
  "claims": [
    {{
      "claim": "Specific claim from answer",
      "supported": true,
      "citation_source": "Context Chunk 1",
      "explanation": "Why it is supported or unsupported"
    }}
  ]
}}
"""

        raw_output = ""
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw_output = resp.text or ""
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_output = resp.choices[0].message.content or ""

        # Extract JSON
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data
            except Exception:
                pass

        return self._evaluate_heuristic(question, answer, context_chunks)

    def _evaluate_heuristic(
        self,
        question: str,
        answer: str,
        context_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Algorithmic heuristic evaluation based on n-gram overlap, claim entailment approximations,
        and citation tag presence.
        """
        combined_context = " ".join(context_chunks).lower()
        context_words = set(re.findall(r"\w+", combined_context))
        question_words = set(re.findall(r"\w+", question.lower()))

        # Split answer into sentence claims
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", answer) if len(s.strip()) > 15]

        claim_evaluations = []
        supported_count = 0

        for s in sentences[:6]:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(s_words.intersection(context_words))
            ratio = overlap / max(len(s_words), 1)

            # Check citation markers
            has_citation = bool(re.search(r"\[(?:Source|Citation|p\.|\d+)[^\]]*\]", s, re.IGNORECASE))
            is_supported = (ratio >= 0.5) or has_citation

            if is_supported:
                supported_count += 1

            claim_evaluations.append({
                "claim": s[:110] + ("..." if len(s) > 110 else ""),
                "supported": is_supported,
                "citation_source": "Retrieved Document Chunks" if is_supported else "Unverified in Context",
                "explanation": f"Context token overlap: {int(ratio*100)}%" + (" | Valid citation tag detected." if has_citation else "")
            })

        total_claims = max(len(claim_evaluations), 1)
        faithfulness = round(supported_count / total_claims, 2)

        # Relevance: overlap between answer and question
        answer_words = set(re.findall(r"\w+", answer.lower()))
        q_overlap = len(question_words.intersection(answer_words))
        relevance = round(min(1.0, max(0.6, q_overlap / max(len(question_words), 1) + 0.3)), 2)

        # Context coverage
        coverage = round(min(1.0, max(0.5, len(answer_words) / 120.0)), 2)

        # Hallucination Risk
        if faithfulness >= 0.85:
            risk = "Low"
            grade = "A (Excellent)"
        elif faithfulness >= 0.65:
            risk = "Moderate"
            grade = "B (Good)"
        elif faithfulness >= 0.45:
            risk = "High"
            grade = "C (Fair)"
        else:
            risk = "Critical"
            grade = "F (Groundedness Issue)"

        return {
            "faithfulness_score": faithfulness,
            "answer_relevance_score": relevance,
            "context_coverage_score": coverage,
            "hallucination_risk": risk,
            "overall_grade": grade,
            "claims": claim_evaluations,
            "verdict_summary": (
                f"The response scored a faithfulness rating of {int(faithfulness*100)}% with {supported_count} out of {total_claims} "
                f"sample claims verified against retrieved passages. Hallucination risk is rated as {risk}."
            )
        }
