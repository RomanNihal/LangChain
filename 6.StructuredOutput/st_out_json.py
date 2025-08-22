from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Review = {
#     "title": "anything",
#     "type": "object",
#     "properties": {
#         "key_themes":{
#             "type": "array",
#             "items":{
#                 "type": "string"
#             },
#             "description": "Write down all the key themes discussed in the review in a list"
#         },
#         "summary": {
#             "type": "string",
#             "description": "A brief summary of the review"
#         },
#         "sentiment": {
#             "type": "string",
#             "enum": ["positive", "negative", "neutral"],
#             "description": "Return sentiment of the review either positive, negative or neutral"
#         },
#         "pros": {
#             "type": ["array", "null"],
#             "items": {
#                 "type": "string"
#             },
#             "description": "Write down all the pros insede a list"
#         },
#         "cons": {
#             "type": ["array", "null"],
#             "items": {
#                 "type": "string"
#             },
#             "description": "Write down all the cons insede a list"
#         },
#         "name": {
#             "type": ["string", "null"],
#             "description": "Write the name of the reviewer"
#         }
#     },
#     "required": ["key_themes", "summary", "sentiment"]
# }

Review = {
  "function_declarations": [
    {
      "name": "Review",
      "description": "Extracts key information from a review.",
      "parameters": {
        "type": "object",
        "properties": {
          "key_themes": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
          },
          "summary": {
            "type": "string",
            "description": "A brief summary of the review"
          },
          "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Return sentiment of the review either positive, negative or neutral"
          },
          "pros": {
            "type": ["array", "null"],
            "items": {
              "type": "string"
            },
            "description": "Write down all the pros inside a list"
          },
          "cons": {
            "type": ["array", "null"],
            "items": {
              "type": "string"
            },
            "description": "Write down all the cons inside a list"
            },
            "name": {
              "type": ["string", "null"],
              "description": "Write the name of the reviewer"
            }
        },
        "required": ["key_themes", "summary", "sentiment"]
      }
    }
  ]
}

st_model = model.with_structured_output(Review)
output = st_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.

Review by Roman Nihal               
""")

# print(output[0]['args'])
print(output)
