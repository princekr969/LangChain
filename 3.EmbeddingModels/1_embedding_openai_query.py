from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# we give parameter dimension, to get that size of vector
# So this generate the embedding of a single query
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension=32)
result = embedding.embed_query("Delhi is the capital of India")
print(str(result))
