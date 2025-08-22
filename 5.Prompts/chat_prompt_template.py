from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    # SystemMessage(content="You are a skilled engineer"),
    # HumanMessage(content="Tell me about LangChain")
    # the above method doesn't work for the placeholders. so the following is used

    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({"domain": "cricket", "topic": "Wicket"})

print(prompt)