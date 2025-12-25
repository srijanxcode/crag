from load_docs import load_documents
from chunking import chunk_text
from embeddings import embed_texts

docs = load_documents("../data/docs")

all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_text(doc))

embeddings = embed_texts(all_chunks)

assert len(all_chunks) == embeddings.shape[0], "Mismatch: chunks vs embeddings"

print("Number of chunks:", len(all_chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding (first 5 values):", embeddings[0][:5])

