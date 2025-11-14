import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

BASE_DIR = Path(__file__).resolve().parent
SPEECH_PATH = BASE_DIR / "speech.txt"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

def build_vectorstore(recreate: bool = False) -> Chroma:
    if not SPEECH_PATH.exists():
        print("ERROR: speech.txt not found.")
        sys.exit(1)
    if recreate and CHROMA_PERSIST_DIR.exists():
        for file in CHROMA_PERSIST_DIR.iterdir():
            file.unlink()
        print("Rebuilding Chroma DB...")
    loader = TextLoader(str(SPEECH_PATH), encoding="utf-8")
    docs = loader.load()

    if not docs:
        raise ValueError("speech.txt is empty.")
    
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Could not create text chunks. Check speech.txt.")

    print(f"Created {len(chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )
    vectordb.persist()
    print("Vectorstore saved to:", CHROMA_PERSIST_DIR)
    return vectordb

def build_qa_chain(vectordb: Chroma) -> RetrievalQA:
    llm = Ollama(model="mistral", temperature=0.0)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

def qa_loop(qa_chain: RetrievalQA):
    print("Type a question, or type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("exited")
            break
        if query.lower() == "rebuild":
            vectordb = build_vectorstore(recreate=True)
            qa_chain = build_qa_chain(vectordb)
            print("Rebuilt DB.")
            continue
        try:
           response = qa_chain.invoke(query)
           answer = response["result"]
           print("\nAI:", answer, "\n")

        except Exception as e:
            print("Error:", e)

def main():
    CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
    vectordb = build_vectorstore(recreate=False)
    qa_chain = build_qa_chain(vectordb)
    qa_loop(qa_chain)

if __name__ == "__main__":
    main()

