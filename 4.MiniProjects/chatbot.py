# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
# from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# pipe = pipeline(
#     task="text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     temperature=0.5,
#     max_new_tokens=20
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# model = ChatHuggingFace(llm=llm)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

chat_history = [
    SystemMessage(content="You are a skilled engineer")
]

while True:
    u_input = input("you: ")
    chat_history.append(HumanMessage(content=u_input))

    if u_input == "exit":
        break

    output = model.invoke(chat_history)
    chat_history.append(AIMessage(content=output.content)) # use the following while using local llm's: ".split("<|assistant|>")[-1].strip()" to trim the output.

    print("AI: ", output.content)

print(chat_history)