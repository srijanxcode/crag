import streamlit as st
import os
import shutil
import numpy as np

from load_docs import load_documents
from chunking import chunk_text
from embeddings import embed_texts
from vector_store import build_index, search
from llm import generate_answer, generate_web_answer
from failure_classifier import classify_failure
from web_search import web_search

# ---------------- CONFIG ----------------
TOP_K = 3
RETRY_K = 6
MAX_RETRIES = 1
MIN_SIMILARITY_THRESHOLD = 0.45
TEMP_DIR = "temp_uploads"
# ---------------------------------------

st.set_page_config(page_title="Failure-Aware CRAG Engine", layout="wide")

st.title("🧠 Failure-Aware CRAG Engine")
st.caption("Document-grounded QA with retry logic & transparent web fallback")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- SESSION STATE ----------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embeddings = None

# ---------------- DOCUMENT INGESTION ----------------
if uploaded_files:
    with st.spinner("Processing uploaded documents..."):

        # Clean temp directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Save uploaded PDFs
        for file in uploaded_files:
            file_path = os.path.join(TEMP_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Load documents (expects directory)
        docs = load_documents(TEMP_DIR)

        # Chunk documents
        chunks = []
        for doc in docs:
            chunks.extend(chunk_text(doc))

        # Embed chunks
        embeddings = embed_texts(chunks)

        # Build FAISS index
        index = build_index(embeddings)

        # Save to session
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

    st.sidebar.success(f"Loaded {len(chunks)} chunks")

# ---------------- QUESTION INPUT ----------------
question = st.text_input("❓ Ask a question")

# ---------------- MAIN PIPELINE ----------------
if st.button("Get Answer") and question and st.session_state.index:

    with st.spinner("Retrieving relevant documents..."):
        query_embedding = embed_texts([question])
        results = search(
            st.session_state.index,
            query_embedding,
            st.session_state.chunks,
            k=TOP_K
        )

    similarities = [r["similarity"] for r in results]
    texts = [r["text"] for r in results]

    max_sim = max(similarities)
    avg_sim = float(np.mean(similarities))

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Similarity", round(max_sim, 3))
    col2.metric("Avg Similarity", round(avg_sim, 3))
    col3.metric("Chunks Retrieved", len(results))

    # ---------------- HARD HALLUCINATION GATE ----------------
    if max_sim < MIN_SIMILARITY_THRESHOLD:
        st.error("❌ No relevant document support found.")
        st.warning("🌐 Falling back to web search")

        web_context = web_search(question)
        web_answer = generate_web_answer(question, web_context)

        st.write(web_answer)
        st.caption("⚠️ This answer was generated from web sources")

        st.stop()

    # ---------------- INITIAL ANSWER ----------------
    context = "\n\n".join(texts[:2])
    answer = generate_answer(question, context)

    failure = classify_failure(max_sim, avg_sim, answer)
    retries = 0

    # ---------------- CRAG RETRY LOGIC ----------------
    if failure in ["WEAK_CONTEXT", "MODEL_UNCERTAIN"] and retries < MAX_RETRIES:
        retries += 1
        st.warning("🔁 Weak context detected — retrying with more evidence")

        retry_results = search(
            st.session_state.index,
            query_embedding,
            st.session_state.chunks,
            k=RETRY_K
        )

        retry_context = "\n\n".join([r["text"] for r in retry_results])
        retry_answer = generate_answer(question, retry_context)

        retry_failure = classify_failure(max_sim, avg_sim, retry_answer)

        if retry_failure == "OK":
            answer = retry_answer
            failure = "OK"

    # ---------------- FINAL DECISION ----------------
    if failure == "OK":
        st.success("✅ Answer (Document-grounded)")
        st.write(answer)
        source = "documents"

    elif failure == "OUT_OF_SCOPE":
        st.warning("🌐 Question outside document scope — using web")

        web_context = web_search(question)
        web_answer = generate_web_answer(question, web_context)

        st.write(web_answer)
        source = "web"

    else:
        st.error("⚠️ Unable to answer confidently")
        st.write(answer)
        source = "documents"

    # ---------------- EXPLAINABILITY ----------------
    with st.expander("🔍 Retrieved Chunks"):
        for i, r in enumerate(results):
            st.markdown(f"**Chunk {i+1} (similarity={round(r['similarity'], 3)})**")
            st.write(r["text"])

    with st.expander("📊 Decision Metrics"):
        st.json({
            "failure_reason": failure,
            "retries_used": retries,
            "answer_source": source
        })
