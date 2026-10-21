import os
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from frontend.components.styles import get_custom_css

# Configure page layout
st.set_page_config(
    page_title="AI Research Paper & PDF Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None

# Backend Health Check & Auto-start Helper
@st.cache_resource
def ensure_backend_running():
    def check():
        try:
            r = requests.get(f"{API_BASE_URL}/", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    if not check():
        try:
            import threading
            import uvicorn
            from backend.main import app as fastapi_app
            
            def run_server():
                uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()

            # Wait briefly for server startup
            import time
            for _ in range(12):
                if check():
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"Failed to auto-start background FastAPI server: {e}")

    return check()

backend_online = ensure_backend_running()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/reading-scientific-sources.png", width=64)
    st.title("Settings & Control")
    
    # Backend status badge
    if backend_online:
        st.success("● Backend API Online (Port 8000)")
    else:
        st.error("● Backend API Offline")
        st.caption("Please run `python run_backend.py` in your terminal.")

    st.markdown("---")
    st.subheader("🤖 Model & Provider")
    provider = st.selectbox(
        "LLM & Embedding Provider",
        options=["gemini", "openai", "local"],
        format_func=lambda x: "Google Gemini 2.5 Flash" if x == "gemini" else ("OpenAI GPT-4o-mini" if x == "openai" else "Local Extractive (Offline Fallback)"),
        index=0
    )

    api_key_input = ""
    if provider != "local":
        default_key = os.getenv("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            f"{provider.capitalize()} API Key",
            type="password",
            value=default_key,
            help="Your API key is used for this session and not logged or stored remotely."
        )

    st.markdown("---")
    st.subheader("⚙️ Retrieval Tuning")
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=15, value=5, help="Number of context passages retrieved per query")
    enable_hybrid = st.checkbox("Enable Hybrid Search (Dense + BM25)", value=True)
    alpha = 0.7
    if enable_hybrid:
        alpha = st.slider("Hybrid Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                          help="1.0 = Pure Semantic Vector Search | 0.0 = Pure Keyword/BM25 Search | 0.7 = Balanced Hybrid")

    st.markdown("---")
    # Quick Library Stats
    docs_list = []
    if backend_online:
        try:
            r = requests.get(f"{API_BASE_URL}/api/documents", timeout=3)
            if r.status_code == 200:
                docs_list = r.json()
        except Exception:
            pass

    total_docs = len(docs_list)
    total_chunks = sum(d.get("num_chunks", 0) for d in docs_list)
    total_pages = sum(d.get("num_pages", 0) for d in docs_list)

    st.metric("Indexed Papers", total_docs)
    st.metric("Total Pages", total_pages)
    st.metric("Vector Chunks", total_chunks)

# ----------------- MAIN HEADER -----------------
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 AI Research Paper & PDF Assistant</h1>
        <p>Intelligent RAG pipeline for research papers, project reports, and PDFs with verbatim citations, cross-paper comparison, and hallucination evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not backend_online:
    st.warning("⚠️ FastAPI backend is currently unreachable. Start it with `python run_backend.py` to enable indexing and querying.")

# ----------------- NAVIGATION TABS -----------------
tab_qa, tab_docs, tab_compare, tab_eval, tab_search, tab_summary = st.tabs([
    "💬 Research Q&A & Citations",
    "📄 Document Hub & Ingestion",
    "⚖️ Multi-Paper Comparison",
    "📊 Answer Evaluation & Metrics",
    "🔍 Hybrid Search Explorer",
    "📑 Executive Summarizer"
])

# ==============================================================
# TAB 1: RESEARCH Q&A WITH CITATIONS
# ==============================================================
with tab_qa:
    st.subheader("Academic Q&A with Page References & Evidence Quotes")
    st.caption("Ask complex questions across single or multiple papers. Every answer is grounded with exact page citations.")

    if not docs_list:
        st.info("💡 No papers are currently indexed. Go to the **📄 Document Hub & Ingestion** tab to upload PDFs or fetch a paper directly from arXiv!")
    else:
        # Document selector
        doc_options = {d["doc_id"]: f"{d['title']} ({d['filename']})" for d in docs_list}
        selected_doc_ids = st.multiselect(
            "Filter Context by Document (leave empty to search all indexed papers):",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x]
        )

        # Question input
        user_query = st.text_input(
            "Research Query / Hypothesis / Question:",
            placeholder="e.g., What attention mechanism is introduced, and what is its computational complexity?",
            key="user_query_input"
        )

        col_q1, col_q2, col_q3 = st.columns([1, 1, 4])
        with col_q1:
            ask_btn = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)
        with col_q2:
            clear_btn = st.button("🗑️ Clear History", use_container_width=True)
            if clear_btn:
                st.session_state.chat_history = []
                st.session_state.last_query_result = None
                st.rerun()

        if ask_btn and user_query.strip():
            with st.spinner("Retrieving hybrid context passages and synthesizing grounded citations..."):
                payload = {
                    "question": user_query.strip(),
                    "doc_ids": selected_doc_ids if selected_doc_ids else None,
                    "top_k": top_k,
                    "hybrid_search": enable_hybrid,
                    "alpha": alpha,
                    "provider": provider,
                    "api_key": api_key_input
                }
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/query", json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.last_query_result = data
                        st.session_state.chat_history.insert(0, data)
                    else:
                        st.error(f"API Error: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

        # Display Last / Active Query Response
        if st.session_state.last_query_result:
            result = st.session_state.last_query_result
            st.markdown("---")
            
            # Response Header with Metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Execution Time", f"{result['execution_time_seconds']}s")
            with m_col2:
                st.metric("Grounded Confidence", f"{int(result['confidence_score']*100)}%")
            with m_col3:
                st.metric("Citations Linked", len(result["citations"]))
            with m_col4:
                if st.button("🧪 Evaluate Faithfulness", use_container_width=True):
                    with st.spinner("Running automated answer evaluation suite..."):
                        eval_payload = {
                            "question": result["question"],
                            "answer": result["answer"],
                            "context_chunks": [c["text"] for c in result["retrieved_chunks"]],
                            "provider": provider,
                            "api_key": api_key_input
                        }
                        try:
                            eval_resp = requests.post(f"{API_BASE_URL}/api/evaluate", json=eval_payload, timeout=30)
                            if eval_resp.status_code == 200:
                                st.session_state.eval_result = eval_resp.json()
                                st.success("Evaluation complete! Navigate to the 📊 Answer Evaluation tab to view the full report.")
                            else:
                                st.error(f"Evaluation error: {eval_resp.text}")
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")

            # Answer Content
            st.markdown("### 📝 Synthesized Academic Answer")
            st.markdown(result["answer"])

            # Citations Section
            if result.get("citations"):
                st.markdown("### 📚 Page References & Evidence Quotes")
                for c in result["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-title">[Source {c['citation_id']}] <b>{c['filename']}</b> &nbsp;•&nbsp; Page {c['page_number']}</span>
                                <span class="citation-badge">Relevance: {int(c['relevance_score']*100)}%</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">Section: {c.get('section_header', 'Body Context')}</div>
                            <div class="citation-quote">"{c['quote']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Context Chunks Inspector Accordion
            with st.expander("🔍 View Raw Retrieved Context Chunks from Vector DB"):
                for idx, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                    st.markdown(f"**Passage {idx} | `{chunk['filename']}` (Page {chunk['page_number']}) - Score: {chunk.get('score', 'N/A')}**")
                    st.text(chunk["text"])
                    st.markdown("---")

# ==============================================================
# TAB 2: DOCUMENT HUB & INGESTION
# ==============================================================
with tab_docs:
    st.subheader("Document Ingestion & Management Hub")
    
    col_up1, col_up2 = st.columns(2)
    
    with col_up1:
        st.markdown("#### 📤 Upload Research Papers & Reports (PDF)")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files:",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("📥 Process & Index Uploaded PDFs", type="primary", disabled=not uploaded_files):
            with st.spinner("Extracting text, preserving page numbers, and generating vector embeddings..."):
                files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                data_payload = {"provider": provider, "api_key": api_key_input}
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/upload", files=files_payload, data=data_payload, timeout=90)
                    if resp.status_code == 200:
                        st.success(resp.json().get("message", "Uploaded successfully!"))
                        st.rerun()
                    else:
                        st.error(f"Upload error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to upload: {str(e)}")

    with col_up2:
        st.markdown("#### 🌐 Fetch Paper Directly from ArXiv")
        st.caption("Provide an arXiv Paper ID or URL (e.g. `1706.03762` or `https://arxiv.org/abs/2312.11805`).")
        arxiv_input = st.text_input("ArXiv ID or URL:", placeholder="e.g., 1706.03762")
        if st.button("⚡ Download & Index from ArXiv", disabled=not arxiv_input.strip()):
            with st.spinner("Fetching paper metadata from arXiv API, downloading PDF, and building vector index..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/arxiv",
                        json={"arxiv_query_or_url": arxiv_input.strip()},
                        params={"provider": provider, "api_key": api_key_input},
                        timeout=90
                    )
                    if resp.status_code == 200:
                        res_data = resp.json()
                        st.success(res_data.get("message", "Paper indexed!"))
                        st.rerun()
                    else:
                        st.error(f"Error fetching arXiv paper: {resp.text}")
                except Exception as e:
                    st.error(f"ArXiv download failed: {str(e)}")

    st.markdown("---")
    st.markdown("#### 📚 Indexed Research Library")
    if not docs_list:
        st.info("No documents currently stored in vector database.")
    else:
        for doc in docs_list:
            with st.container():
                d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns([3, 1, 1, 1, 1])
                with d_col1:
                    st.markdown(f"**📄 {doc.get('title', doc['filename'])}**")
                    st.caption(f"Filename: `{doc['filename']}` • Uploaded: {doc.get('upload_time', 'N/A')}")
                with d_col2:
                    st.write(f"📖 {doc['num_pages']} Pages")
                with d_col3:
                    st.write(f"🧩 {doc['num_chunks']} Chunks")
                with d_col4:
                    st.write(f"💾 {doc['file_size_kb']} KB")
                with d_col5:
                    if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}", use_container_width=True):
                        try:
                            d_resp = requests.delete(f"{API_BASE_URL}/api/documents/{doc['doc_id']}")
                            if d_resp.status_code == 200:
                                st.success("Deleted!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.markdown("---")

# ==============================================================
# TAB 3: MULTI-PAPER COMPARISON STUDIO
# ==============================================================
with tab_compare:
    st.subheader("⚖️ Multi-Document Comparative Matrix")
    st.caption("Compare methodologies, datasets, experimental results, and limitations across 2 or more research papers side-by-side.")

    if len(docs_list) < 2:
        st.info("Please index at least 2 research papers in the **📄 Document Hub** to perform comparative analysis.")
    else:
        doc_options = {d["doc_id"]: f"{d['title']} ({d['filename']})" for d in docs_list}
        selected_compare_ids = st.multiselect(
            "Select 2 or more papers to compare:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            default=list(doc_options.keys())[:2]
        )

        st.markdown("##### Comparison Dimensions")
        default_aspects = [
            "Research Objective & Problem Statement",
            "Proposed Methodology & Architecture",
            "Datasets & Evaluation Benchmarks",
            "Key Empirical Results & Metrics",
            "Reported Limitations & Future Work"
        ]
        chosen_aspects = st.multiselect(
            "Select or customize comparison dimensions:",
            options=default_aspects,
            default=default_aspects
        )

        if st.button("🔬 Generate Comparative Analysis", type="primary", disabled=len(selected_compare_ids) < 2):
            with st.spinner("Synthesizing multi-paper comparative matrix..."):
                payload = {
                    "doc_ids": selected_compare_ids,
                    "aspects": chosen_aspects,
                    "provider": provider,
                    "api_key": api_key_input
                }
                try:
                    comp_resp = requests.post(f"{API_BASE_URL}/api/compare", json=payload, timeout=90)
                    if comp_resp.status_code == 200:
                        comp_data = comp_resp.json()
                        st.markdown("### 📊 Meta-Review & Comparative Breakdown")
                        st.markdown(comp_data["executive_synthesis"])
                        
                        # Download button
                        st.download_button(
                            "📥 Download Comparison Report (Markdown)",
                            data=comp_data["executive_synthesis"],
                            file_name="paper_comparison_report.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"Comparison error: {comp_resp.text}")
                except Exception as e:
                    st.error(f"Comparison request failed: {e}")

# ==============================================================
# TAB 4: ANSWER EVALUATION & HALLUCINATION INSPECTOR
# ==============================================================
with tab_eval:
    st.subheader("📊 RAG Evaluation Suite: Faithfulness & Hallucination Metrics")
    st.caption("Quantitative verification measuring whether the generated answers are strictly grounded in retrieved evidence.")

    eval_data = st.session_state.eval_result

    if not eval_data:
        st.info("💡 Run a query in the **💬 Research Q&A** tab and click '**🧪 Evaluate Faithfulness**' to inspect the live metrics here, or manually run an evaluation below.")
        
        with st.expander("📝 Manual Evaluation Sandbox"):
            manual_q = st.text_input("Question:", value="What is the attention mechanism?")
            manual_ans = st.text_area("Generated Answer to evaluate:", value="The Transformer model uses Scaled Dot-Product Attention and Multi-Head Attention to compute representations without recurrence.")
            manual_ctx = st.text_area("Grounding Context Passage:", value="We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys of dimension dk, and values of dimension dv.")
            if st.button("Run Manual Evaluation"):
                with st.spinner("Evaluating..."):
                    payload = {
                        "question": manual_q,
                        "answer": manual_ans,
                        "context_chunks": [manual_ctx],
                        "provider": provider,
                        "api_key": api_key_input
                    }
                    try:
                        resp = requests.post(f"{API_BASE_URL}/api/evaluate", json=payload, timeout=30)
                        if resp.status_code == 200:
                            st.session_state.eval_result = resp.json()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        # Metrics Display
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            st.metric("Faithfulness (Groundedness)", f"{int(eval_data['faithfulness_score'] * 100)}%")
        with col_e2:
            st.metric("Answer Relevance", f"{int(eval_data['answer_relevance_score'] * 100)}%")
        with col_e3:
            st.metric("Context Coverage", f"{int(eval_data['context_coverage_score'] * 100)}%")
        with col_e4:
            risk = eval_data.get("hallucination_risk", "Low")
            badge_class = "badge-low-risk" if risk == "Low" else ("badge-mod-risk" if risk == "Moderate" else "badge-high-risk")
            st.markdown(f"**Hallucination Risk:**<br><span class='{badge_class}'>{risk} Risk</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 📋 Evaluator Verdict: **{eval_data.get('overall_grade', 'A')}**")
        st.write(eval_data.get("verdict_summary", ""))

        # Claim Verification Breakdown
        claims = eval_data.get("claims", [])
        if claims:
            st.markdown("#### 🔬 Claim-by-Claim Verification Breakdown")
            for c in claims:
                supported_badge = "✅ Supported" if c["supported"] else "❌ Unsupported / Hallucination Risk"
                bg_color = "#f0fdf4" if c["supported"] else "#fef2f2"
                st.markdown(
                    f"""
                    <div style="background: {bg_color}; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div><b>Claim:</b> "{c['claim']}"</div>
                        <div style="margin-top: 4px; font-size: 0.85rem;">
                            <b>Status:</b> {supported_badge} &nbsp;|&nbsp; 
                            <b>Evidence Source:</b> {c.get('citation_source', 'Context')}
                        </div>
                        <div style="margin-top: 4px; font-size: 0.85rem; color: #475569;">
                            <b>Detail:</b> {c.get('explanation', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==============================================================
# TAB 5: HYBRID SEARCH EXPLORER
# ==============================================================
with tab_search:
    st.subheader("🔍 Dense & Sparse Hybrid Search Diagnostics")
    st.caption("Inspect raw chunk embeddings, cosine distances, and BM25 lexical token frequencies directly.")

    search_q = st.text_input("Test Retrieval Query:", placeholder="e.g., self-attention computational complexity")
    s_col1, s_col2 = st.columns([1, 4])
    with s_col1:
        search_btn = st.button("🔎 Run Search Diagnostics", type="primary", use_container_width=True)

    if search_btn and search_q.strip():
        with st.spinner("Searching vector index..."):
            try:
                s_resp = requests.post(
                    f"{API_BASE_URL}/api/search",
                    params={
                        "query": search_q.strip(),
                        "top_k": top_k,
                        "hybrid": enable_hybrid,
                        "alpha": alpha
                    },
                    timeout=15
                )
                if s_resp.status_code == 200:
                    search_results = s_resp.json().get("results", [])
                    st.write(f"Found **{len(search_results)}** matching passages.")
                    for item in search_results:
                        with st.container():
                            st.markdown(f"##### 📄 `{item['filename']}` — Page {item['page_number']} [{item.get('section_header', 'Section')}]")
                            st.caption(f"Combined Score: **{item['score']}** | Dense Similarity: **{item.get('dense_score', 'N/A')}** | Sparse Lexical: **{item.get('sparse_score', 'N/A')}**")
                            st.info(item["text"])
                            st.markdown("---")
                else:
                    st.error(f"Search error: {s_resp.text}")
            except Exception as e:
                st.error(f"Failed to query: {e}")

# ==============================================================
# TAB 6: EXECUTIVE SUMMARIZER
# ==============================================================
with tab_summary:
    st.subheader("📑 Paper Executive Summarizer & Key Takeaways")
    st.caption("Synthesize comprehensive summaries, methodology breakdowns, and empirical highlights for any indexed paper.")

    if not docs_list:
        st.info("No documents indexed. Upload a paper first.")
    else:
        doc_options = {d["doc_id"]: f"{d['title']} ({d['filename']})" for d in docs_list}
        sum_doc_id = st.selectbox(
            "Choose a paper to summarize:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x]
        )
        
        sum_type = st.radio(
            "Summary Focus:",
            options=["comprehensive", "executive", "methodology", "key_takeaways"],
            format_func=lambda x: {
                "comprehensive": "Comprehensive Overview & Analysis",
                "executive": "Executive Brief (High-level abstract & impact)",
                "methodology": "Technical Methodology & Architecture",
                "key_takeaways": "Key Empirical Findings & Takeaways"
            }[x],
            horizontal=True
        )

        if st.button("⚡ Generate Structured Summary", type="primary"):
            with st.spinner("Analyzing document structure and synthesizing executive summary..."):
                sum_payload = {
                    "doc_id": sum_doc_id,
                    "summary_type": sum_type,
                    "provider": provider,
                    "api_key": api_key_input
                }
                try:
                    s_resp = requests.post(f"{API_BASE_URL}/api/summarize", json=sum_payload, timeout=60)
                    if s_resp.status_code == 200:
                        sum_data = s_resp.json()
                        st.markdown(f"## {sum_data.get('title', 'Document Summary')}")
                        st.markdown(sum_data["summary"])
                    else:
                        st.error(f"Summary error: {s_resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
