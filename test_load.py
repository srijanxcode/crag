from load_docs import load_documents

docs = load_documents("../data/docs")
print("Number of docs:", len(docs))
print(docs[0][:300])
