from langsmith import Client    
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

prompt = Client().pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
