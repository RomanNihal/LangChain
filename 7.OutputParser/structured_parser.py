from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=50
# )

# model = ChatHuggingFace(llm=llm)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

schema = [
    ResponseSchema(name="fact_1", description="fact no 1 about the topic"),
    ResponseSchema(name="fact_2", description="fact no 2 about the topic"),
    ResponseSchema(name="fact_3", description="fact no 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.invoke({"topic":"cricket"})

# output = model.invoke(prompt)

# print(parser.parse(output.content))

chain = template | model | parser

output = chain.invoke({"topic":"football"})

print(output)
