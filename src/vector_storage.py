import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Load Data (since it is in JSON Lines format, lines=True)
print("Reading movie details...")
movies_df = pd.read_json('IMDB_movie_details.json', lines=True)

# 2. Filter for those with a summary (plot_synopsis)
# Let's start with the first 1000 movies so as not to exhaust your computer
movies_sample = movies_df[movies_df['plot_synopsis'].notna()].head(1000)

docs = []
for _, row in movies_sample.iterrows():
    # Let's get the column name safely
    m_id = str(row['movie_id'])
    # If movie_name is missing, let's use movie_id as the name
    m_title = row.get('movie_name', m_id) 
    
    docs.append(Document(
        page_content=row['plot_synopsis'],
        metadata={"movie_id": m_id, "title": m_title}
    ))

# 3. Split Text into Chunks (small chunks that the agent can read)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 4. Embedding Model (HuggingFace)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Create the Vector Database (ChromaDB)
print(f"{len(split_docs)} text chunks are being vectorized... (This may take a few minutes)")
vector_db = Chroma.from_documents(
    documents=split_docs, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Successfully completed! The 'chroma_db' folder is ready as the project's memory.")