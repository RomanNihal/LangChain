from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "Sundar Pichai is the CEO of Google"
documents = [
    "My name is Roman Nihal",
    "I am a CSE graduate",
    "I am currently learning LangChain"
]

output = embedding.embed_documents(documents)

print(str(output))