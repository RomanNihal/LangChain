from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# schema
class Review(BaseModel):
    key_themes: List[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Return sentiment of the review either positive, negative or neutral")
    pros: Optional[List[str]] = Field(default=None, description="Write down all the pros insede a list")
    cons: Optional[List[str]] = Field(default=None, description="Write down all the cons insede a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

st_model = model.with_structured_output(Review)
output = st_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.

Review by Roman Nihal               
""")

print(output)
