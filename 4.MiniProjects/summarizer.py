from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

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

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", [" Beginner- Friendly", "Technical", "Code-oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)" , "Long (detailed explanation)"])


template = load_prompt('template.json')


if st.button('summarize'):
    chain = template | model
    # fill the placeholders
    output = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(output.content)