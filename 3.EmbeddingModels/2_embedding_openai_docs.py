from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Delhi is the capital of India",
    "Kolkata is the capital of west bengal",
    "Paris is the capital of France"
]

# we give parameter dimension, to get that size of vector
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension=32)
result = embedding.embed_query("Delhi is the capital of India")
print(str(result))
