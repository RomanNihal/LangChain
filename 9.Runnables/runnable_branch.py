from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text into 10 words \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>10, prompt2 | model | parser), # type: ignore
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

output = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(output)
