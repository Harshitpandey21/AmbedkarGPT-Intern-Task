# AmbedkarGPT-Intern-Task

<h1 align="center">AmbedkarGPT – RAG Powered CLI Assistant</h1> <p align="center"> Retrieval-Augmented Generation · ChromaDB · LangChain · Ollama Mistral 7B </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" /> <img src="https://img.shields.io/badge/LangChain-0.1.x-green?style=flat-square" /> <img src="https://img.shields.io/badge/Mistral-7B-red?style=flat-square" /> <img src="https://img.shields.io/badge/ChromaDB-VectorStore-orange?style=flat-square" /> <img src="https://img.shields.io/badge/Embeddings-HuggingFace-black?style=flat-square" /> </p>

# 📌 Overview

AmbedkarGPT is a small Retrieval-Augmented Generation (RAG) system designed to answer questions based on a short excerpt from Dr.B.R.Ambedkar’s writings.

It demonstrates a complete local RAG workflow using:
* LangChain for orchestration
* ChromaDB as the vector database
* HuggingFace sentence embeddings
* Ollama (Mistral 7B) for LLM inference

The application runs entirely offline and provides a clean CLI interface for querying the text.

# 📁 Project Structure
```
AmbedkarGPT-Intern-Task
    ├── main.py            
    ├── requirements.txt   
    ├── speech.txt         
    └── README.md          
```

# How It Works

1. Loads the provided source text (speech.txt)

2. Splits the content into smaller chunks

3. Converts chunks into embeddings using a HuggingFace model

4. Stores them in a persistent Chroma vector database

5. Retrieves relevant chunks when the user enters a question

6. Passes retrieved context into Mistral 7B running locally via Ollama

7. Produces a grounded, context-aware answer

# Installation & Setup
1. Clone the repository
```
git clone <https://github.com/Harshitpandey21/AmbedkarGPT-Intern-Task.git>
cd AmbedkarGPT-Intern-Task
```

2. Create and activate a virtual environment(for mac)
```
python3 -m venv venv
source venv/bin/activate
```
for windows:-
```
python -m venv venv
venv\Scripts\activate
```
3. Install Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4. Install Ollama and pull Mistral 7B
Ollama handles the local LLM.
```
curl -fsSL https://ollama.ai/install.sh | sh
```
Then pull:
```
ollama pull mistral
```
5. Running the Application
Start the CLI:
```
python main.py
```
# 📦 requirements.txt
```
langchain>=0.0.320
langchain-community
chromadb>=0.3.27
sentence-transformers>=2.2.2
transformers>=4.30.0
torch>=2.0.0
ollama>=0.0.7
huggingface-hub>=0.14.1
ftfy
regex
tqdm
```
