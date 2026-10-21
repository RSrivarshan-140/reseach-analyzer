import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional
from backend.config import UPLOADS_DIR

class ArxivClient:
    def __init__(self, download_dir: Path = UPLOADS_DIR):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def extract_arxiv_id(self, query: str) -> Optional[str]:
        """
        Extracts clean arXiv ID from URL or raw string.
        Examples:
        - 1706.03762
        - 1706.03762v5
        - https://arxiv.org/abs/1706.03762
        - https://arxiv.org/pdf/1706.03762.pdf
        - arXiv:1706.03762
        """
        query = query.strip()
        match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def fetch_and_download(self, arxiv_id_or_url: str) -> Dict[str, Any]:
        """
        Fetches metadata from arXiv API and downloads the paper PDF.
        """
        arxiv_id = self.extract_arxiv_id(arxiv_id_or_url)
        if not arxiv_id:
            raise ValueError(f"Could not identify a valid arXiv ID from '{arxiv_id_or_url}'. Please provide an ID like '1706.03762'.")

        # 1. Fetch metadata via arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "AI-Research-Assistant/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", namespace)

        if entry is None or entry.find("atom:title", namespace) is None:
            raise ValueError(f"No paper found on arXiv for ID {arxiv_id}")

        title = entry.find("atom:title", namespace).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", namespace).text.strip().replace("\n", " ")
        published = entry.find("atom:published", namespace).text[:10] if entry.find("atom:published", namespace) is not None else ""
        
        authors = []
        for author in entry.findall("atom:author", namespace):
            name_node = author.find("atom:name", namespace)
            if name_node is not None and name_node.text:
                authors.append(name_node.text.strip())

        # Clean filename
        clean_title = re.sub(r"[^\w\s-]", "", title)[:45].strip().replace(" ", "_")
        filename = f"arxiv_{arxiv_id}_{clean_title}.pdf"
        target_path = self.download_dir / filename

        # 2. Download PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(pdf_req, timeout=30) as pdf_resp, open(target_path, "wb") as out_file:
            out_file.write(pdf_resp.read())

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "pdf_url": pdf_url,
            "local_path": str(target_path),
            "filename": filename
        }
