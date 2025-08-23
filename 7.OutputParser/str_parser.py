from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# pipe = pipeline(
#     task="text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     temperature=0.5,
#     max_new_tokens=50
# )

# llm = HuggingFacePipeline(pipeline=pipe)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 point summary on the following text:/n {text}",
    input_variables=["text"]
)

# prompt1 = template1.invoke({"topic": "sun"})

# output1 = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": output1.content})

# output2 = model.invoke(prompt2)

# print(output2.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

output = chain.invoke({"topic": "sun"})

print(output)