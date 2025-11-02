# 🌟 LCEL (LangChain Expression Language)

## 🧠 What is LCEL?
**LCEL (LangChain Expression Language)** is a **simpler and faster** way to connect LangChain components like prompts, models, and parsers.  
It replaces older, more verbose classes like `SequentialChain` with a clean **pipe (`|`) syntax**.

---

## ⚙️ What Does It Do?
LCEL allows you to:
- **Compose** multiple LangChain steps easily  
- **Run** them synchronously, asynchronously, or in **streaming mode**  
- **Automatically pass** outputs between steps  

---

## 💡 Benefits
✅ Cleaner, more readable code  
✅ **Built-in streaming** and **async support**  
✅ **Faster execution** (can run parts concurrently)  
✅ Works consistently with all LangChain components  

---

## 🧩 Example Comparison

### 🆚 Old Way vs New Way

```python
# OLD WAY — Using SequentialChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt = PromptTemplate(
    input_variables=["text"],
    template="Translate this sentence to French: {text}"
)
llm = OpenAI()

chain = SequentialChain(chains=[prompt, llm], input_variables=["text"])
result = chain.run("I love programming.")
print(result)


# NEW WAY — Using LCEL
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

prompt = ChatPromptTemplate.from_template("Translate this sentence to French: {text}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser  # LCEL composition
result = chain.invoke({"text": "I love programming."})
print(result)
