from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

# Step 1: Create the Hugging Face pipeline object.
# This is where you specify the task, model, and generation parameters.
pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.5,
    max_new_tokens=100
)

# Step 2: Pass the pipeline object to the HuggingFacePipeline class.
# This class's job is just to be a wrapper around the existing pipeline.
llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(llm=llm)

output = model.invoke("Who is the CEO of google?")

print(output.content)
