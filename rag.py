from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

query = "What should be done if gas concentration and pressure are both high during maintenance?"

results = vectordb.similarity_search(query, k=3)

print("\nTOP RESULTS:\n")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)