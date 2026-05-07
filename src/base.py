# base.py
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)