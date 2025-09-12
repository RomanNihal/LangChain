from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")
    text: str
 
parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative \n{feedback} \n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

classifier = prompt | model | parser

parser_str = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write an appropriate response as a chatbot to this positive feedback - Simple and acknowledging \n{feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="Write an appropriate response as a chatbot for this negative feedback - Simple and acknowledging \n{feedback}",
    input_variables=["feedback"]
)

chain1 = prompt1 | model | parser_str

chain2 = prompt2 | model | parser_str

branch = RunnableBranch(
    (lambda x: x.sentiment == "positive", RunnableLambda(lambda x: chain1.invoke({"feedback": x.text}))), # type: ignore
    (lambda x: x.sentiment == "negative", RunnableLambda(lambda x: chain2.invoke({"feedback": x.text}))), # type: ignore
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier | branch

output = chain.invoke({"feedback": "This is not a good movie"})

print(output)