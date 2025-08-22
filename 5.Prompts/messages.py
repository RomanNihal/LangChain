from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

message = [
    SystemMessage(content="You are a skilled engineer"),
    HumanMessage(content="Tell me about LangChain")
]

output = model.invoke(message)

message.append(AIMessage(content=output.content))

print(message)