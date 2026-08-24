from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension=300)

documents = [
    "Virat Kohli is an Indian international cricketer and one of the most successful batsmen in modern cricket.",
    "He has represented the Indian cricket team in all three formats of the game.",
    "Virat Kohli is known for his aggressive batting style, fitness, and consistency.",
    "He has played for Royal Challengers Bengaluru in the Indian Premier League.",
    "Kohli has received several awards for his outstanding performances in international cricket."
]

query = 'tell me about virat kohli'

doc_embeddings = embed_documents(documents)
query_embedding = embedding.embed_query(query)
# Cosine similarity measures how similar the two vectors are based on their direction.
scores  = cosine_similarity([query_embedding], doc_embeddings)[0]

# enumerate(scores) adds an index to each score
# list(enumerate(scores)) Converting it to a list
# key=lambda x: x[1] Sort these (index, score) pairs according to the score.
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
# Give me the index and similarity score of the document that is most similar to my query.

print(query) 
print(documents[index]) 
print("Similarity score is:", score)