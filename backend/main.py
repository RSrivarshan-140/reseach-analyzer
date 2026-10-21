import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import UPLOADS_DIR, BACKEND_HOST, BACKEND_PORT
from backend.models.schemas import (
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    CompareRequest,
    CompareResponse,
    EvaluateRequest,
    EvaluationResult,
    SummarizeRequest,
    SummaryResponse,
    ArxivFetchRequest,
    DeleteDocumentResponse
)
from backend.core.pdf_processor import PDFProcessor
from backend.core.vector_store import VectorStoreManager
from backend.core.llm_service import LLMService
from backend.core.evaluator import AnswerEvaluator
from backend.core.arxiv_client import ArxivClient

app = FastAPI(
    title="AI Research Paper & PDF Assistant API",
    description="High-performance backend API for RAG, academic PDF parsing, citations, and evaluation.",
    version="1.0.0"
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
pdf_processor = PDFProcessor()
vector_store = VectorStoreManager()
llm_service = LLMService()
evaluator = AnswerEvaluator()
arxiv_client = ArxivClient()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "AI Research Paper & PDF Assistant",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/documents", response_model=List[DocumentMetadata])
def list_documents():
    """Returns all indexed research papers and documents."""
    return vector_store.list_all_documents()

@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    provider: str = Form("gemini"),
    api_key: Optional[str] = Form(None)
):
    """
    Uploads and indexes one or more PDF files into the vector database with page-aware chunking.
    """
    processed_docs = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        doc_id = str(uuid.uuid4())[:8]
        safe_filename = f"{doc_id}_{file.filename}"
        target_path = UPLOADS_DIR / safe_filename

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            metadata_dict, chunks = pdf_processor.extract_document(target_path, doc_id=doc_id)
            metadata_dict["filename"] = file.filename
            metadata_dict["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Persist to ChromaDB vector store
            vector_store.add_chunks(chunks, provider=provider, api_key=api_key)
            vector_store.save_doc_metadata(doc_id, metadata_dict)
            processed_docs.append(metadata_dict)
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")

    return {
        "message": f"Successfully processed {len(processed_docs)} document(s).",
        "documents": processed_docs
    }

@app.post("/api/arxiv")
def fetch_arxiv_paper(
    request: ArxivFetchRequest,
    provider: str = "gemini",
    api_key: Optional[str] = None
):
    """
    Downloads and indexes a research paper directly from arXiv by ID or URL.
    """
    try:
        paper_info = arxiv_client.fetch_and_download(request.arxiv_query_or_url)
        local_path = Path(paper_info["local_path"])
        doc_id = str(uuid.uuid4())[:8]

        metadata_dict, chunks = pdf_processor.extract_document(local_path, doc_id=doc_id)
        metadata_dict["title"] = paper_info["title"]
        metadata_dict["filename"] = paper_info["filename"]
        metadata_dict["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_dict["summary_preview"] = paper_info["summary"][:400]

        vector_store.add_chunks(chunks, provider=provider, api_key=api_key)
        vector_store.save_doc_metadata(doc_id, metadata_dict)

        return {
            "message": f"Successfully fetched and indexed arXiv paper: {paper_info['title']}",
            "arxiv_info": paper_info,
            "document": metadata_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
def query_research_assistant(request: QueryRequest):
    """
    Executes hybrid RAG retrieval and generates an answer strictly grounded in citations.
    """
    # 1. Retrieve relevant chunks
    chunks = vector_store.search(
        query=request.question,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
        hybrid=request.hybrid_search,
        alpha=request.alpha,
        provider=request.provider or "gemini",
        api_key=request.api_key
    )

    # 2. Synthesize answer with citations
    rag_result = llm_service.generate_rag_response(
        question=request.question,
        retrieved_chunks=chunks,
        provider=request.provider or "gemini",
        api_key=request.api_key
    )

    return QueryResponse(
        question=request.question,
        answer=rag_result["answer"],
        citations=rag_result["citations"],
        retrieved_chunks=chunks,
        execution_time_seconds=rag_result["execution_time_seconds"],
        confidence_score=rag_result["confidence_score"]
    )

@app.post("/api/search")
def search_chunks(
    query: str,
    top_k: int = 5,
    hybrid: bool = True,
    alpha: float = 0.7,
    doc_ids: Optional[str] = None
):
    """
    Direct semantic & hybrid retrieval explorer.
    """
    id_list = [d.strip() for d in doc_ids.split(",")] if doc_ids else None
    results = vector_store.search(
        query=query,
        doc_ids=id_list,
        top_k=top_k,
        hybrid=hybrid,
        alpha=alpha
    )
    return {"query": query, "results": results, "count": len(results)}

@app.post("/api/compare", response_model=CompareResponse)
def compare_research_papers(request: CompareRequest):
    """
    Compares two or more research papers across core academic dimensions.
    """
    if len(request.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Document comparison requires at least 2 document IDs.")

    all_metadata = vector_store.load_all_metadata()
    docs_chunks = {}

    for doc_id in request.doc_ids:
        doc_info = all_metadata.get(doc_id)
        doc_name = doc_info["title"] if doc_info else doc_id
        chunks = vector_store.get_document_chunks(doc_id, limit=8)
        docs_chunks[doc_name] = chunks

    result = llm_service.compare_documents(
        docs_chunks=docs_chunks,
        aspects=request.aspects,
        provider=request.provider or "gemini",
        api_key=request.api_key
    )

    return CompareResponse(**result)

@app.post("/api/evaluate", response_model=EvaluationResult)
def evaluate_rag_answer(request: EvaluateRequest):
    """
    Evaluates faithfulness, relevance, context coverage, and hallucination risk for an answer.
    """
    eval_result = evaluator.evaluate_response(
        question=request.question,
        answer=request.answer,
        context_chunks=request.context_chunks,
        provider=request.provider or "gemini",
        api_key=request.api_key
    )
    return EvaluationResult(**eval_result)

@app.post("/api/summarize", response_model=SummaryResponse)
def summarize_document(request: SummarizeRequest):
    """
    Produces executive summary and structured findings for a specific document.
    """
    all_metadata = vector_store.load_all_metadata()
    doc_meta = all_metadata.get(request.doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = vector_store.get_document_chunks(request.doc_id, limit=15)
    summary_data = llm_service.summarize_paper(
        doc_metadata=doc_meta,
        chunks=chunks,
        summary_type=request.summary_type,
        provider=request.provider or "gemini",
        api_key=request.api_key
    )

    return SummaryResponse(**summary_data)

@app.delete("/api/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str):
    """
    Deletes a document from the vector store and metadata registry.
    """
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document could not be deleted or was not found.")
    return DeleteDocumentResponse(
        doc_id=doc_id,
        status="success",
        message="Document and indexed chunks successfully removed."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
