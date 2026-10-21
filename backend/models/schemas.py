from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    title: Optional[str] = "Untitled Document"
    num_pages: int
    num_chunks: int
    file_size_kb: float
    upload_time: str
    summary_preview: Optional[str] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    section_header: Optional[str] = None
    score: Optional[float] = None

class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    hybrid_search: bool = True
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for dense retrieval vs sparse BM25")
    api_key: Optional[str] = None
    provider: Optional[str] = "gemini" # 'gemini' or 'openai' or 'local'

class CitationItem(BaseModel):
    citation_id: int
    doc_id: str
    filename: str
    page_number: int
    section_header: Optional[str] = None
    quote: str
    relevance_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationItem]
    retrieved_chunks: List[DocumentChunk]
    execution_time_seconds: float
    confidence_score: float

class CompareRequest(BaseModel):
    doc_ids: List[str]
    aspects: Optional[List[str]] = None
    api_key: Optional[str] = None
    provider: Optional[str] = "gemini"

class DocumentComparisonItem(BaseModel):
    aspect: str
    findings: Dict[str, str] # doc_name -> finding
    synthesis: str

class CompareResponse(BaseModel):
    doc_ids: List[str]
    doc_names: List[str]
    comparisons: List[DocumentComparisonItem]
    executive_synthesis: str
    execution_time_seconds: float

class EvaluateRequest(BaseModel):
    question: str
    answer: str
    context_chunks: List[str]
    api_key: Optional[str] = None
    provider: Optional[str] = "gemini"

class ClaimVerification(BaseModel):
    claim: str
    supported: bool
    citation_source: Optional[str] = None
    explanation: str

class EvaluationResult(BaseModel):
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0 indicating how strictly the answer is grounded in the retrieved context.")
    answer_relevance_score: float = Field(description="Score between 0.0 and 1.0 indicating how directly the answer addresses the question.")
    context_coverage_score: float = Field(description="Score between 0.0 and 1.0 indicating how well key context facts were utilized.")
    hallucination_risk: str = Field(description="'Low', 'Moderate', or 'High'")
    overall_grade: str = Field(description="'A (Excellent)', 'B (Good)', 'C (Fair)', 'F (Groundedness Issue)'")
    claims: List[ClaimVerification] = []
    verdict_summary: str

class SummarizeRequest(BaseModel):
    doc_id: str
    summary_type: str = "comprehensive" # "comprehensive", "methodology", "executive", "key_takeaways"
    api_key: Optional[str] = None
    provider: Optional[str] = "gemini"

class SummaryResponse(BaseModel):
    doc_id: str
    filename: str
    title: str
    summary: str
    key_findings: List[str]
    methodology_overview: Optional[str] = None
    limitations: Optional[str] = None

class ArxivFetchRequest(BaseModel):
    arxiv_query_or_url: str

class DeleteDocumentResponse(BaseModel):
    doc_id: str
    status: str
    message: str
