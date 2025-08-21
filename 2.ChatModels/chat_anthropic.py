from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-sonnet-4-20250514', temperature=0, max_tokens_to_sample=10)

output = model.invoke("Who is the CEO of google?")

print(output.content)