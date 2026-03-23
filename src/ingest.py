from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"

def ingest_pdf(pdf_path : str):
    """
    Takes a PDF File path , chunks it , embeds it and store ir in a chromadb
    """

    print(f"Loading PDF:{pdf_path}")

    #step 1 :load the pdf
    #pypdfloader reads the pdf and returns a list of documemt objects
    #each document = a page from pdf

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF")

    #step 2 : Split intp chunks
    #we split in smaller chunks because:
    # 1.llm have token limits
    #2. smaller chunks = more precise retrieval
    #example chunk size = 500 : each chunk is ~500 characters
    #chunk overlap = 100: chunks share 100 chars with neighbours so we dont loase context at boundaries

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators=["\n\n","\n","."," "]
    )
    chunks = splitter.split_documents(documents)

    print(f"split into {len(chunks)} chunks")

    #step3:create embeddings
    #an embedding converts text into a list of numbers(vectors)
    #similar meaning = similar numbers = can be searched
    #we will use free huggingface model - no api key is needed

    embeddings = HuggingFaceEmbeddings(
        model_name ="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("Cleared existing ChromaDB database.")
    #step 4 : store in chromadb
    # chroma.from_documents does 3 things automatically:
    #1.takes each chunk
    #2.converts it to an embedding vector
    #3.stores chunk + vector in local chromadb database

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding = embeddings,
        persist_directory="chroma_db"
    )

    print(f"Done!{len(chunks)} chunks stored in Chromabd.")
    print("Database saved to: chroma_db/")

    return vectorstore

# this block only runs when you execute this file directory 
#python src/ingest.py

if __name__ == "__main__":
    pdf_path = "C:/1SKILL COMBACK/rag-projects/research-rag/data/attention.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error:Could not find {pdf_path}")
        print(f"Looking in: {os.getcwd()}")
        print("Please add a PDF file to the data/ folder")
    else:
        ingest_pdf(pdf_path)