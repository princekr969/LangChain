from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"
docs = [
    "Delhi is the capital of India",
    "Kolkata is the capital of west bengal",
    "Paris is the capital of France"
]
query_vector = embedding.embed_query(text)
docs_vector = embedding.embed_query(text)
print(str(query_vector))
print(str(docs_vector))