MAX_RETRIES = 1
RETRY_K = 6
from web_search import web_search
from llm import generate_web_answer
from failure_classifier import classify_failure
from load_docs import load_documents
from chunking import chunk_text
from embeddings import embed_texts
from vector_store import build_index, search
from llm import generate_answer

print("🚀 test_rag.py started")

DOCS_PATH = "../data/docs"
TOP_K = 3
MIN_SIMILARITY_THRESHOLD = 0.45

# 1. Load documents
docs = load_documents(DOCS_PATH)
print("📄 Documents loaded:", len(docs))

# 2. Chunk documents
chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))
print("🧩 Total chunks:", len(chunks))

# 3. Embed chunks
chunk_embeddings = embed_texts(chunks)
print("📐 Embeddings shape:", chunk_embeddings.shape)

# 4. Build FAISS index
index = build_index(chunk_embeddings)
print("🗂️ FAISS index built")

# 5. Query
question = "What is Chichen Itza?"
print("❓ Question:", question)

query_embedding = embed_texts([question])
print("🔍 Query embedded")

# 6. Retrieve
results = search(index, query_embedding, chunks, k=TOP_K)
print("📊 Retrieval results:", len(results))

similarities = [r["similarity"] for r in results]
texts = [r["text"] for r in results]

max_similarity = max(similarities)
avg_similarity = sum(similarities) / len(similarities)

print("\n--- RETRIEVAL STATS ---")
print("Max similarity:", round(max_similarity, 3))
print("Avg similarity:", round(avg_similarity, 3))

# 7. Hard gate
if max_similarity < MIN_SIMILARITY_THRESHOLD:
    print("\nANSWER:")
    print("I don't know (no relevant document support)")
    exit()

# 8. Context (keep small)
context = "\n\n".join(texts[:2])
print("🧠 Context prepared")

# 9. Generate answer
answer = generate_answer(question, context)

failure_reason = classify_failure(
    max_similarity=max_similarity,
    avg_similarity=avg_similarity,
    answer=answer
)

print("\nANSWER:")
print(answer)

print("\n🧠 FAILURE CLASSIFICATION:")
print(failure_reason)

# ---------- WEB FALLBACK ----------
if failure_reason == "OUT_OF_SCOPE":
    print("\n🌐 WEB FALLBACK TRIGGERED")

    web_context = web_search(question)

    if not web_context.strip():
        print("\nANSWER (Web-generated):")
        print("No relevant information found on the web.")
    else:
        web_answer = generate_web_answer(question, web_context)

        print("\nANSWER (Web-generated):")
        print(web_answer)

        print("\n⚠️ NOTE:")
        print("This answer was generated using web search, not your documents.")

