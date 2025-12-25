from load_docs import load_documents
from chunking import chunk_text

docs = load_documents("../data/docs")
chunks = chunk_text(docs[0])
print("Total chunks:", len(chunks))
print(chunks[0][:300])

