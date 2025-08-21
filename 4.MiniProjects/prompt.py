from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header("Research Summarizer")

pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.5,
    max_new_tokens=50
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(llm=llm)

user_input = st.text_input("Enter your prompt")

if st.button('summarize'):
    output = model.invoke(user_input)
    st.write(output.content)