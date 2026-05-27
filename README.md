## Step-by-Step Installation and Execution Guide

Follow these instructions sequentially to set up the project and the data pipeline from scratch on a new machine.

### Step 1: Clone the Repository & Setup Virtual Environment

First, clone the repository and create an isolated Python environment:

```bash
git clone [https://github.com/rojdaineurope/multi-agentic-spoiler-app.git](https://github.com/rojdaineurope/multi-agentic-spoiler-app.git)
cd multi-agentic-spoiler-app
```

# Create virtual environment

```bash
python -m venv venv
```

# Activate (Windows)

venv\Scripts\activate

# Activate (macOS/Linux)

source venv/bin/activate

### Step 2: Install Dependencies

Install all required libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
(Alternatively, install manually: pip install langchain-chroma chromadb langchain-huggingface groq python-dotenv pandas scikit-learn streamlit)
```

### Step 3: Download the Dataset

Download the raw IMDb Spoiler Dataset from Kaggle: IMDB Spoiler Dataset

Extract the downloaded .zip file.

Move the dataset files (e.g., IMDB_reviews.json, IMDB_movie_details.json) into the data/ directory of the project.

### Step 4: Configure Environment Variables

Create a .env file in the root directory of the project and insert your Groq API key to enable the Llama-3 model:

Kod snippet'i

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Data Preprocessing

Run the preprocessing script. This will clean the raw IMDb dataset (removing punctuation and noise) and output a new file named cleaned_reviews.csv:

```bash
python src/preprocess.py
Step 6: Initialize the ChromaDB Vector Database
Run the vector storage script. This step vectorizes the cleaned data using the HuggingFace embedding model and creates the local ./chroma_db directory required for the Retriever Agent:
```

```bash
python src/vector_storage.py
```

### Step 7: Launch the Application

Once the vector database is ready, start the main Multi-Agent Streamlit dashboard:

```bash
streamlit run app2.py
```
