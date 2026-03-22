from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function = embeddings
)

count = vector_store._collection.count()

print(f"Total chunks in chromadb:{count}")

results = vector_store._collection.get(limit=3)

print("\n printing First 3 chunks")
for i, doc in enumerate(results["documents"]):
    print(f"\nChunk{i+1}:")
    print(doc[:300])
    print("-"*50)