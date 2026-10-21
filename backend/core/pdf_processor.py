import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_document(self, file_path: Path, doc_id: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extracts metadata, pages, and chunks from a PDF file.
        Returns (metadata_dict, list_of_chunks).
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]

        reader = PdfReader(str(file_path))
        num_pages = len(reader.pages)
        pdf_metadata = reader.metadata or {}

        # Extract title from metadata or fallback to first non-empty lines
        title = None
        if pdf_metadata.title and str(pdf_metadata.title).strip():
            title = str(pdf_metadata.title).strip()

        pages_text: List[Dict[str, Any]] = []
        raw_full_text = []

        current_section = "Introduction / Front Matter"
        section_pattern = re.compile(
            r"^(?:\d+\.?\s+)?(Abstract|Introduction|Related\s+Work|Background|Methodology|Method|Architecture|System\s+Design|Experimental\s+Setup|Experiments|Results|Discussion|Conclusion|References|Acknowledgements)\b",
            re.IGNORECASE
        )

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            pages_text.append({"page_number": page_idx, "text": text})
            raw_full_text.append(text)

        combined_text = "\n".join(raw_full_text)

        # Infer title if missing
        if not title:
            first_page_lines = [l.strip() for l in pages_text[0]["text"].split("\n") if len(l.strip()) > 3]
            if first_page_lines:
                title = first_page_lines[0][:120]
            else:
                title = file_path.stem.replace("_", " ").title()

        # Chunk pages while preserving exact page numbers and section headers
        chunks: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page_info in pages_text:
            p_num = page_info["page_number"]
            p_text = page_info["text"]

            if not p_text.strip():
                continue

            # Update section header if any detected in this page
            lines = p_text.split("\n")
            for line in lines[:8]: # check top lines of page
                match = section_pattern.search(line.strip())
                if match:
                    current_section = match.group(0).strip().title()
                    break

            page_chunks = self._chunk_page_text(p_text, p_num, doc_id, file_path.name, current_section, global_chunk_idx)
            chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        metadata = {
            "doc_id": doc_id,
            "filename": file_path.name,
            "title": title,
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "file_size_kb": round(file_path.stat().st_size / 1024, 2),
            "summary_preview": (combined_text[:400] + "...") if len(combined_text) > 400 else combined_text
        }

        return metadata, chunks

    def _chunk_page_text(
        self,
        text: str,
        page_number: int,
        doc_id: str,
        filename: str,
        section_header: str,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """
        Splits text of a specific page into overlapping chunks.
        """
        chunks = []
        # Split on sentence boundaries where possible
        sentences = re.split(r"(?<=[.?!])\s+", text)
        
        current_chunk_sentences = []
        current_length = 0
        local_idx = 0

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
                
            sent_len = len(sentence_clean)
            if current_length + sent_len > self.chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences).strip()
                chunks.append({
                    "chunk_id": f"{doc_id}_p{page_number}_c{start_index + local_idx}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": start_index + local_idx,
                    "text": chunk_str,
                    "section_header": section_header
                })
                local_idx += 1

                # Retain overlap sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk_sentences = overlap_sentences
                current_length = overlap_len

            current_chunk_sentences.append(sentence_clean)
            current_length += sent_len

        # Remaining chunk
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences).strip()
            if len(chunk_str) > 20: # ignore trivial noise
                chunks.append({
                    "chunk_id": f"{doc_id}_p{page_number}_c{start_index + local_idx}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": start_index + local_idx,
                    "text": chunk_str,
                    "section_header": section_header
                })

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Normalize ligatures, excessive spacing, and page break artifacts.
        """
        # Replace common PDF ligatures
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
        # Remove null characters and control characters
        text = text.replace("\x00", "")
        # Normalize multiple spaces and non-breaking spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize excessive newlines (keep max 2 for paragraph separation)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
