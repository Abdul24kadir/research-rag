from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"]="false"

def load_vectorstore():
    """
    Loads the existing ChromaDB database from disk.
    We use the same embedding model as ingestion —
    this is critical. If you use a different model,
    the search vectors won't match and results will be wrong.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    print(f"Loaded chromadb with {vectorstore._collection.count()} chunks")
    return vectorstore

def format_docs(docs):
    """
    as retriever returns documents , we will send this docs in single string / passing to llm as context ,also retuens source info for citations"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectorstore):
    """
    Builds a modern LCEL retrieval chain
    LCEL uses the | pipe operator to connect components-
    think of it like pipeline where data flows left to right
    """
    #LLm stepup
    
    endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=450,
    temperature=0.3,
    task="text-generation"
    )
    

    llm = ChatHuggingFace(llm = endpoint)
    #Prompt template
    prompt_template = """You are a helpful research assistant.
    Use ONLY the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the provided document."
Do not make up answers.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context','question']
    )

    #Retriever 

    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3}
    )

    #LCEL chain
    #This is the modern way to build chains in langchain
    #Read it left to right like a pipeline:
    # 1. RunnablePassthrough() — passes the question through unchanged
    # 2. retriever — searches ChromaDB for relevant chunks
    # 3. format_docs — formats chunks into a single context string
    # 4. prompt — fills in {context} and {question}
    # 5. llm — generates the answer
    # 6. StrOutputParser() — converts LLM output to plain string

    chain = (
        {
           "context":retriever|format_docs,
           "question":RunnablePassthrough() 
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain , retriever

def ask_question(chain , retriever, question:str):
    """
    Asks a question and prints the answer with citations
    """
    print(f"\n Question:{question}")
    print("-"*50)
    #Get the answer 
    answer = chain.invoke(question)
    print(f"Answer:{answer}")

    #Get citations seperately - retrieve same chunks again
    source_docs = retriever.invoke(question)
    print("\nSources Used:")
    for i,doc in enumerate(source_docs):
        page = doc.metadata.get("page",None)
        page_display = page + 1 if page is not None else "unkown"
        print(f"[{i+1}] page {page_display}:{doc.page_content[:150]}....")
    
    return answer , source_docs

if __name__=="__main__":
    #load the database
    vectorstore = load_vectorstore()

    #build the chain
    chain , retriever = build_qa_chain(vectorstore)

    #Test questions
    questions = [
        "what is the main contribution of this paper",
        "what is the attention mechanism?",
        "what were the BLEU scores achieved?"
    ]

    for question in questions:
        ask_question(chain,retriever,question)
        print("\n"+"="*60 + "\n")