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

def ingest_pdf(pdf_path: str, chroma_path: str = "chroma_db"):
    """
    Takes a PDF file path, chunks it, embeds it,
    stores in ChromaDB at chroma_path.
    """
    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print("Cleared existing ChromaDB.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
        collection_name="rag_collection"
    )

    print(f"Done! {len(chunks)} chunks stored in ChromaDB.")
    print(f"Database saved to: {chroma_path}")

    return vectorstore


if __name__ == "__main__":
    pdf_path = "data/samples/attention.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        print(f"Looking in: {os.getcwd()}")
    else:
        ingest_pdf(pdf_path)