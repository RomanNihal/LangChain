from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=16)

documents = [
    "My name is Roman Nihal",
    "I am a CSE graduate",
    "I am currently learning LangChain"
]

output = embedding.embed_documents(documents)

print(str(output))