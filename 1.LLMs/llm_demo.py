from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

gpt = OpenAI(model='gpt-3.5-turbo-instruct')

output = gpt.invoke("Who is the CEO of google")\

print(output)