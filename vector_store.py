import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(index, query_embedding, texts, k=3):
    D, I = index.search(query_embedding, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        similarity = 1 / (1 + dist)   # L2 distance → similarity
        results.append({
            "text": texts[idx],
            "similarity": similarity
        })

    return results
