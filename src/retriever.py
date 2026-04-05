# src/retriever.py

from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "false"


def clean_output(text: str) -> str:
    """
    Strips repeated Q&A that Mistral sometimes generates.
    Keeps only the first answer.
    """
    return re.split(r'\[/INST\]|Question:|Human:', text)[0].strip()


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    print(f"Loaded ChromaDB with {vectorstore._collection.count()} chunks")
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_llm():
    """
    Mistral 7B Instruct v0.2 via featherless-ai provider.
    Uses ChatHuggingFace + HuggingFaceEndpoint — proper LangChain style.
    featherless-ai is the only provider that supports this model currently.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip(),
        max_new_tokens=256,
        temperature=0.1,
        provider="featherless-ai"
    )
    return ChatHuggingFace(llm=endpoint)


def build_qa_chain(vectorstore):
    """
    Builds LCEL retrieval chain using Mistral 7B v0.2.
    ChatHuggingFace + HuggingFaceEndpoint — proper LangChain style.
    """
    llm = build_llm()

    prompt = PromptTemplate.from_template("""You are a helpful research assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the provided document."
Do not make up answers. Answer directly and concisely.

Context:
{context}

Question: {question}

Answer:""")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_output)
    )

    return chain, retriever


def ask_question(chain, retriever, question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    source_docs = retriever.invoke(question)
    print("\nSources Used:")
    for i, doc in enumerate(source_docs):
        page = doc.metadata.get("page", None)
        page_display = page + 1 if page is not None else "unknown"
        print(f"  [{i+1}] Page {page_display}: {doc.page_content[:150]}...")
    return answer, source_docs


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chain, retriever = build_qa_chain(vectorstore)
    questions = [
        "What is the main contribution of this paper?",
        "What is the attention mechanism?",
        "What were the BLEU scores achieved?"
    ]
    for question in questions:
        ask_question(chain, retriever, question)
        print("\n" + "=" * 60 + "\n")