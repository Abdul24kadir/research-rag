# src/hybrid_retriever.py

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "false"


def load_vectorstore_and_chunks(base_dir: str = None):
    """
    Loads ChromaDB AND returns raw chunks.
    base_dir — pass the directory where chroma_db lives.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # go up one level if we're inside src/
        if os.path.basename(base_dir) == "src":
            base_dir = os.path.dirname(base_dir)

    chroma_path = os.path.join(base_dir, "chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    raw = vectorstore._collection.get(
        include=["documents", "metadatas"]
    )

    all_chunks = [
        Document(
            page_content=text,
            metadata=meta if meta else {}
        )
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    print(f"Loaded {len(all_chunks)} chunks from {chroma_path}")
    return vectorstore, all_chunks

def get_similarity_results(vectorstore, question: str, k: int = 3) -> List[Document]:
    """
    Retriever 1 — Basic cosine similarity.
    Finds semantically similar chunks.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever.invoke(question)


def get_hybrid_results(vectorstore, all_chunks: List[Document], question: str, k: int = 3) -> List[Document]:
    """
    Retriever 2 — MMR + BM25 hybrid.
    MMR = diverse results from vector search.
    BM25 = exact keyword matching.
    EnsembleRetriever merges both using Reciprocal Rank Fusion.
    """
    # MMR vector retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )

    # BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = k

    # combine with equal weights
    ensemble = EnsembleRetriever(
        retrievers=[mmr_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble.invoke(question)


def rerank_documents(
    question: str,
    documents: List[Document],
    top_k: int = 3
) -> Tuple[List[Document], List[float]]:
    """
    Cross Encoder Reranker.
    Reads query + chunk TOGETHER as a pair.
    More accurate than vector similarity alone.
    Returns top_k docs and their scores.
    """
    if not documents:
        return [], []

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[question, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    top_docs = [doc for score, doc in scored[:top_k]]
    top_scores = [float(score) for score, doc in scored[:top_k]]

    return top_docs, top_scores