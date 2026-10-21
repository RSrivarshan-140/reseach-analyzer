import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from backend.models.schemas import CitationItem, QueryResponse, DocumentComparisonItem, CompareResponse, SummaryResponse

class LLMService:
    def __init__(self):
        pass

    def generate_rag_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        provider: str = "gemini",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a research-grade response strictly grounded in retrieved chunks,
        accompanied by explicit page citations and evidence quotes.
        """
        start_time = time.time()
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant passages were found in the selected documents matching your query. Please try rephrasing or uploading relevant PDF research papers.",
                "citations": [],
                "confidence_score": 0.0,
                "execution_time_seconds": round(time.time() - start_time, 2)
            }

        # Build context prompt with numbered references
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            block = (
                f"--- [SOURCE {i}] ---\n"
                f"Document: {chunk['filename']}\n"
                f"Page: {chunk['page_number']}\n"
                f"Section: {chunk.get('section_header', 'N/A')}\n"
                f"Content: {chunk['text']}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert AI Research Assistant and Academic Peer Reviewer. "
            "Your task is to answer the researcher's question accurately, strictly grounded in the provided document sources.\n\n"
            "CRITICAL CITATION RULES:\n"
            "1. Every factual statement must cite its source using the format [Source X, p. Y], where X is the source number and Y is the page number.\n"
            "2. Whenever mentioning specific metrics, theorems, equations, datasets, or experimental results, quote the specific numbers or findings.\n"
            "3. If different papers or pages present differing perspectives or conflicting evidence, explicitly contrast them.\n"
            "4. Do NOT speculate or invent information not supported by the sources.\n"
            "5. After your comprehensive explanation, include an 'Evidence Breakdown' bulleted section summarizing which source and page supports which finding."
        )

        user_prompt = (
            f"RESEARCH CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"Please provide an in-depth, rigorous research answer adhering strictly to the citation rules."
        )

        active_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        answer_text = ""

        if provider == "gemini" and active_key:
            try:
                answer_text = self._call_gemini(system_prompt, user_prompt, active_key)
            except Exception as e:
                print(f"[Warning] Gemini call failed ({e}), falling back to extractive synthesis.")
                answer_text = self._extractive_synthesis(question, retrieved_chunks)
        elif provider == "openai" and active_key:
            try:
                answer_text = self._call_openai(system_prompt, user_prompt, active_key)
            except Exception as e:
                print(f"[Warning] OpenAI call failed ({e}), falling back to extractive synthesis.")
                answer_text = self._extractive_synthesis(question, retrieved_chunks)
        else:
            answer_text = self._extractive_synthesis(question, retrieved_chunks)

        # Build structured citations from retrieved chunks
        citations: List[Dict[str, Any]] = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Take representative quote snippet
            words = chunk["text"].split()
            quote_snippet = " ".join(words[:45]) + ("..." if len(words) > 45 else "")
            citations.append({
                "citation_id": i,
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "section_header": chunk.get("section_header", "General"),
                "quote": quote_snippet,
                "relevance_score": chunk.get("score", 0.85)
            })

        execution_time = round(time.time() - start_time, 2)
        
        # Calculate heuristic confidence score based on top retrieval score and context presence
        top_score = retrieved_chunks[0].get("score", 0.8)
        confidence = round(min(0.98, max(0.65, float(top_score))), 2)

        return {
            "answer": answer_text,
            "citations": citations,
            "confidence_score": confidence,
            "execution_time_seconds": execution_time
        }

    def compare_documents(
        self,
        docs_chunks: Dict[str, List[Dict[str, Any]]],
        aspects: Optional[List[str]] = None,
        provider: str = "gemini",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs multi-document comparative analysis across specified academic aspects.
        """
        start_time = time.time()
        
        if not aspects:
            aspects = [
                "Research Objective & Problem Statement",
                "Methodology & Proposed Architecture",
                "Datasets & Experimental Benchmarks",
                "Key Quantitative Results & Breakthroughs",
                "Limitations & Open Research Questions"
            ]

        doc_names = list(docs_chunks.keys())
        context_parts = []
        for doc_name, chunks in docs_chunks.items():
            combined_excerpt = "\n".join([f"[p. {c['page_number']}] {c['text'][:400]}" for c in chunks[:6]])
            context_parts.append(f"=== DOCUMENT: {doc_name} ===\n{combined_excerpt}\n")

        full_context = "\n".join(context_parts)

        system_prompt = (
            "You are a Senior Academic Meta-Reviewer. Your job is to produce a rigorous, structured comparison between "
            "multiple research papers or technical reports based on their content."
        )

        aspects_list_str = "\n".join([f"- {a}" for a in aspects])
        user_prompt = (
            f"DOCUMENTS UNDER COMPARISON:\n{full_context}\n\n"
            f"Please conduct an in-depth comparative evaluation across these dimensions:\n{aspects_list_str}\n\n"
            "Format your output in clear markdown with:\n"
            "1. A side-by-side comparative breakdown for each dimension.\n"
            "2. A summary synthesis highlighting key architectural or empirical trade-offs between the papers."
        )

        active_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        comparison_text = ""

        if provider == "gemini" and active_key:
            try:
                comparison_text = self._call_gemini(system_prompt, user_prompt, active_key)
            except Exception as e:
                comparison_text = self._extractive_comparison(docs_chunks, aspects)
        elif provider == "openai" and active_key:
            try:
                comparison_text = self._call_openai(system_prompt, user_prompt, active_key)
            except Exception as e:
                comparison_text = self._extractive_comparison(docs_chunks, aspects)
        else:
            comparison_text = self._extractive_comparison(docs_chunks, aspects)

        # Build structured comparison items
        structured_comparisons = []
        for aspect in aspects:
            findings = {}
            for doc_name, chunks in docs_chunks.items():
                first_text = chunks[0]["text"] if chunks else "No text extracted."
                findings[doc_name] = f"Detailed under {doc_name} (p. {chunks[0]['page_number'] if chunks else 1})"
            structured_comparisons.append({
                "aspect": aspect,
                "findings": findings,
                "synthesis": f"Comparative evaluation conducted across {len(doc_names)} documents."
            })

        return {
            "doc_ids": list(docs_chunks.keys()),
            "doc_names": doc_names,
            "comparisons": structured_comparisons,
            "executive_synthesis": comparison_text,
            "execution_time_seconds": round(time.time() - start_time, 2)
        }

    def summarize_paper(
        self,
        doc_metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        summary_type: str = "comprehensive",
        provider: str = "gemini",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an executive research paper summary with key takeaways and methodology breakdown.
        """
        combined_text = "\n\n".join([f"[Page {c['page_number']} - {c.get('section_header', 'Section')}] {c['text']}" for c in chunks[:12]])

        system_prompt = (
            "You are a Research Director summarizing academic papers for busy scientists. "
            "Extract the core technical contributions, methodology, empirical findings, and limitations."
        )

        user_prompt = (
            f"PAPER TITLE: {doc_metadata.get('title', 'Untitled')}\n"
            f"FILENAME: {doc_metadata.get('filename')}\n\n"
            f"PAPER CONTENT SAMPLES:\n{combined_text}\n\n"
            f"Provide a structured summary ({summary_type}) including:\n"
            "1. Executive Overview (2-3 paragraphs)\n"
            "2. Key Findings & Contributions (Bulleted list with page references)\n"
            "3. Methodology & System Architecture\n"
            "4. Limitations & Future Work"
        )

        active_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        summary_text = ""

        if provider == "gemini" and active_key:
            try:
                summary_text = self._call_gemini(system_prompt, user_prompt, active_key)
            except Exception:
                summary_text = self._extractive_summary(doc_metadata, chunks)
        elif provider == "openai" and active_key:
            try:
                summary_text = self._call_openai(system_prompt, user_prompt, active_key)
            except Exception:
                summary_text = self._extractive_summary(doc_metadata, chunks)
        else:
            summary_text = self._extractive_summary(doc_metadata, chunks)

        return {
            "doc_id": doc_metadata.get("doc_id", ""),
            "filename": doc_metadata.get("filename", ""),
            "title": doc_metadata.get("title", ""),
            "summary": summary_text,
            "key_findings": [
                f"Extracted from {len(chunks)} contextual chunks across {doc_metadata.get('num_pages', 1)} pages.",
                f"Primary topic: {doc_metadata.get('title', 'Research Study')}.",
                f"Page-level citations indexed for interactive queries."
            ],
            "methodology_overview": "Extracted from indexed sections.",
            "limitations": "Refer to discussion and conclusion sections."
        }

    def _call_gemini(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n{user_prompt}"
        )
        return response.text or "No response generated by model."

    def _call_openai(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content or "No response generated."

    def _extractive_synthesis(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Extractive synthesis fallback when no external API key is configured.
        Extracts top sentences directly matching question keywords, attributing exact page numbers.
        """
        q_tokens = set(re.findall(r"\w+", question.lower()))
        scored_sentences = []

        for i, chunk in enumerate(chunks, 1):
            sentences = re.split(r"(?<=[.?!])\s+", chunk["text"])
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 25:
                    continue
                s_tokens = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(q_tokens.intersection(s_tokens))
                if overlap > 0:
                    scored_sentences.append({
                        "sentence": s_clean,
                        "overlap": overlap,
                        "source_idx": i,
                        "doc": chunk["filename"],
                        "page": chunk["page_number"],
                        "section": chunk.get("section_header", "General")
                    })

        scored_sentences.sort(key=lambda x: x["overlap"], reverse=True)
        top_sentences = scored_sentences[:6]

        if not top_sentences:
            return (
                f"Based on the indexed passages from **{chunks[0]['filename']}** (Page {chunks[0]['page_number']}), "
                f"the document discusses:\n\n> \"{chunks[0]['text'][:350]}...\" [Source 1, p. {chunks[0]['page_number']}]\n\n"
                f"*Tip: Configure your Gemini or OpenAI API key in the sidebar for full conversational generation and cross-paper synthesis.*"
            )

        output_lines = [
            f"### Synthesized Answer from Document Sources\n",
            f"Based on analysis of **{len(chunks)}** relevant passages retrieved from the uploaded documents:\n"
        ]

        for item in top_sentences:
            output_lines.append(
                f"- **{item['sentence']}** [Source {item['source_idx']}: *{item['doc']}*, p. {item['page']}, Section: {item['section']}]"
            )

        output_lines.append(
            "\n\n#### Evidence & Context Alignment\n"
            "The retrieved passages directly match key terms in your query. "
            "To unlock advanced reasoning and synthesis across complex equations and figures, enter your Gemini API Key in the sidebar."
        )

        return "\n".join(output_lines)

    def _extractive_comparison(self, docs_chunks: Dict[str, List[Dict[str, Any]]], aspects: List[str]) -> str:
        lines = ["# Comparative Analysis Matrix\n"]
        for aspect in aspects:
            lines.append(f"### {aspect}\n")
            for doc_name, chunks in docs_chunks.items():
                snippet = chunks[0]["text"][:250] if chunks else "No content available."
                page = chunks[0]["page_number"] if chunks else 1
                lines.append(f"- **{doc_name}** (p. {page}): {snippet}...\n")
        return "\n".join(lines)

    def _extractive_summary(self, doc_metadata: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        lines = [
            f"# Executive Summary: {doc_metadata.get('title', 'Document')}\n",
            f"**File:** `{doc_metadata.get('filename')}` | **Pages:** {doc_metadata.get('num_pages')} | **Indexed Chunks:** {doc_metadata.get('num_chunks')}\n",
            "### Overview & Abstract\n"
        ]
        if chunks:
            lines.append(f"> {chunks[0]['text'][:450]}...\n")
        lines.append("\n### Key Extracted Sections\n")
        for c in chunks[:5]:
            lines.append(f"- **Page {c['page_number']} [{c.get('section_header', 'Section')}]:** {c['text'][:180]}...")
        return "\n".join(lines)
