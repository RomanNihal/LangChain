from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano', temperature=0.2, max_completion_tokens=10)

output = model.invoke("Who is the CEO of google?")

print(output.content)