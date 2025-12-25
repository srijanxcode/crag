from load_docs import load_documents
from chunking import chunk_text
from embeddings import embed_texts
from vector_store import build_index, search

docs = load_documents("../data/docs")
chunks = chunk_text(docs[0])
embeddings = embed_texts(chunks)

index = build_index(embeddings)
query_emb = embed_texts(["Chichen Itza"])

results = search(index, query_emb, chunks)
for r in results:
    print(r["similarity"], r["text"][:100])

