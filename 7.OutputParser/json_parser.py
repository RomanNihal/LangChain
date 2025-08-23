from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
    task="text-generation",
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, and address of a anime character\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.format() # or we can use .invoke({})

# output = model.invoke(prompt)

# print(parser.parse(output.content))

chain = template | model | parser

output = chain.invoke({})

print(output)