import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Ensure we identify our requests; allow override via env or .env
os.environ.setdefault(
    "USER_AGENT", "langgraph-course/1.0 (+https://github.com/MiuTue/LangGraphStudy)"
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)

doc_splits = text_splitter.split_documents(docs_list)
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     embedding=OllamaEmbeddings(model="nomic-embed-text"),
#     collection_name="advanced-agentic-rag",
#     persist_directory="./.chroma_db"
# )

retriever = Chroma(
    collection_name="advanced-agentic-rag",
    persist_directory="./.chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
).as_retriever()
