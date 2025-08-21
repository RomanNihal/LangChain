from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership." ,
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting ecords.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about ms dhoni"

doc_embed = embedding.embed_documents(documents)
query_embed = embedding.embed_query(query)

scores = cosine_similarity([query_embed], doc_embed)[0]

index, score = (sorted(enumerate(scores), key=lambda x:x[1])[-1])

print(query)
print(documents[index])
print("Similarity score:", score)