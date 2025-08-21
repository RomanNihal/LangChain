from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=16)

output = embedding.embed_query("Sundar Pichai is the CEO of Google")

print(str(output))