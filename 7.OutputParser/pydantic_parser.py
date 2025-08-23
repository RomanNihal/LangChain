from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

class character(BaseModel):
    name: str = Field(description="Name of the character")
    age: int = Field(description="Age of the character")
    ability: str = Field(description="Special ability of the character")

parser = PydanticOutputParser(pydantic_object=character)

template = PromptTemplate(
    template="Generate the name, age, and ability of a character from {name} anime\n{format_instruction}",
    input_variables=["name"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt = template.invoke({"Jujutsu Kaisen"})
# print(prompt)

chain = template | model | parser

output = chain.invoke({"Jujutsu Kaisen"})

print(output)