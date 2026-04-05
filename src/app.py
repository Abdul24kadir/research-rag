# app.py

import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import chromadb
import shutil
import uuid
from dotenv import load_dotenv

from ingest import ingest_pdf
from retriever import build_llm, format_docs, load_vectorstore, build_qa_chain
from hybrid_retriever import (
    load_vectorstore_and_chunks,
    get_similarity_results,
    get_hybrid_results,
    rerank_documents
)

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# global state
vectorstore_global = None
all_chunks_global = None
chain_global = None
retriever_global = None


def reset_chroma():
    """
    Clears ChromaDB from memory cache.
    Taken from your old app.py — handles Windows file locks.
    """
    global chain_global, retriever_global
    chain_global = None
    retriever_global = None

    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except:
        pass

    # delete relative chroma_db (handles when CWD is project root)
    if os.path.exists("chroma_db"):
        try:
            shutil.rmtree("chroma_db")
        except:
            pass

    gc.collect()


def reset_chain():
    """
    Clears all global state + deletes chroma_db using absolute path.
    Two-step reset — memory first, then disk.
    """
    global chain_global, retriever_global, vectorstore_global, all_chunks_global

    chain_global = None
    retriever_global = None
    all_chunks_global = None

    if vectorstore_global is not None:
        try:
            vectorstore_global._client._system.stop()
        except:
            pass
        vectorstore_global = None

    gc.collect()

    # delete using absolute path — works regardless of CWD
    chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
        except:
            pass


def process_pdf(pdf_file):
    """
    Handles PDF upload.
    Runs reset_chroma first (memory cache),
    then reset_chain (full state + disk),
    then ingests the new PDF.
    """
    # run both resets like your old app.py
    reset_chroma()

    global vectorstore_global, all_chunks_global, chain_global, retriever_global

    if pdf_file is None:
        return "Please upload a PDF file first."

    try:
        reset_chain()

        # use absolute path for data directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = os.path.basename(pdf_file.name)
        unique_name = f"{uuid.uuid4()}_{filename}"
        dest_path = os.path.join(data_dir, unique_name)
        shutil.copy(pdf_file.name, dest_path)

        # ingest with absolute chroma path
        chroma_path = os.path.join(base_dir, "chroma_db")
        ingest_pdf(dest_path, chroma_path=chroma_path)

        # load vectorstore + chunks for both retrievers
        vectorstore_global, all_chunks_global = load_vectorstore_and_chunks(base_dir)

        # also build basic chain for fallback
        chain_global, retriever_global = build_qa_chain(vectorstore_global)

        return "PDF processed! Ask a question to compare retrievers."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def format_chunks_for_display(docs, scores=None):
    if not docs:
        return "No chunks retrieved."

    output = ""

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", None)
        page_display = page + 1 if page is not None else "unknown"

        score_str = f"  |  Score: {scores[i]:.3f}" if scores else ""

        # 🔥 FIX: force new line before chunk title
        output += "\n" + "─" * 60 + "\n"
        output += f"\nChunk {i+1}  —  Page {page_display}{score_str}\n\n"

        output += f"{doc.page_content[:300]}\n"

    return output.strip()


def compare_retrievers(question):
    """
    Runs basic similarity AND hybrid MMR+BM25.
    Shows results side by side — no answer generated yet.
    """
    global vectorstore_global, all_chunks_global

    if vectorstore_global is None:
        msg = "Please upload and process a PDF first."
        return msg, msg

    if not question.strip():
        msg = "Please enter a question."
        return msg, msg

    try:
        # Retriever 1 — basic similarity
        sim_docs = get_similarity_results(
            vectorstore_global, question, k=3
        )
        sim_display = format_chunks_for_display(sim_docs)

        # Retriever 2 — hybrid MMR + BM25
        hybrid_docs = get_hybrid_results(
            vectorstore_global, all_chunks_global, question, k=3
        )
        hybrid_display = format_chunks_for_display(hybrid_docs)

        return sim_display, hybrid_display

    except Exception as e:
        error = f"Error: {str(e)}"
        return error, error


def generate_answer(question):
    """
    Combines both retriever results,
    reranks with cross encoder,
    generates final answer.
    """
    global vectorstore_global, all_chunks_global

    if vectorstore_global is None:
        return "Please upload and process a PDF first.", ""

    if not question.strip():
        return "Please enter a question.", ""

    try:
        # get results from both retrievers
        sim_docs = get_similarity_results(
            vectorstore_global, question, k=3
        )
        hybrid_docs = get_hybrid_results(
            vectorstore_global, all_chunks_global, question, k=3
        )

        # combine and deduplicate
        seen = set()
        all_candidates = []
        for doc in sim_docs + hybrid_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_candidates.append(doc)

        # rerank combined candidates
        reranked_docs, rerank_scores = rerank_documents(
            question, all_candidates, top_k=3
        )
        reranked_display = format_chunks_for_display(
            reranked_docs, rerank_scores
        )

        # generate final answer — with stop instruction
        context = format_docs(reranked_docs)
        prompt_text = f"""You are a helpful research assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the provided document."
Do not make up answers.
Answer the question directly and stop. Do not generate follow-up questions.

Context:
{context}

Question: {question}

Answer (stop after answering):"""

        import re
        llm = build_llm()
        result = llm.invoke(prompt_text)
        final_answer = result.content if hasattr(result, 'content') else str(result)
        final_answer = re.split(r'\[/INST\]|Question:', final_answer)[0].strip()
        return reranked_display, final_answer

    except Exception as e:
        error = f"Error: {str(e)}"
        return error, error


def clear_all():
    """
    Runs both resets and clears all UI fields.
    Same pattern as your old app.py clear_all.
    """
    reset_chroma()
    reset_chain()
    return "Cleared! Upload a new PDF to continue.", "", "", "", ""


# -------------------------------------------------------
# Gradio UI
# -------------------------------------------------------
with gr.Blocks(
    title="AI Research Assistant",
    theme=gr.themes.Base(),
    css="""
    body {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e2e8f0;
    }

    .gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 20px 40px !important;
}
    /* HEADER */
    .header {
        text-align: center;
        padding: 30px 0 10px;
    }

    .header h1 {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header p {
        color: #94a3b8;
        font-size: 14px;
    }

    /* CARD STYLE */
    .glass {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        transition: 0.3s;
    }

    .glass:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.7);
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* BUTTONS */
    button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    button.primary {
        background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
        border: none !important;
    }

    button.secondary {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
    }

    /* SCROLL AREA */
    .scroll {
        max-height: 420px;
        overflow-y: auto;
        padding-right: 6px;
    }

    /* CHAT STYLE ANSWER */
    textarea {
        background: #020617 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 14px;
    }

    footer { display: none !important; }
    """
) as app:

    # HEADER
    gr.HTML("""
    <div class="header">
        <h1>🤖 AI Research Assistant</h1>
        <p>Upload PDFs • Compare retrieval • Get grounded answers</p>
    </div>
    """)

    # TOP SECTION
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="glass"):
                gr.HTML("<div class='section-title'>📂 Upload Document</div>")

                pdf_input = gr.File(file_types=[".pdf"])
                upload_btn = gr.Button("⚡ Process PDF", variant="primary")
                upload_status = gr.Textbox(show_label=False, interactive=False)

                clear_btn = gr.Button("♻ Reset", variant="secondary")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="glass"):
                gr.HTML("<div class='section-title'>💬 Ask Question</div>")

                question_input = gr.Textbox(
                    placeholder="Ask anything about the paper...",
                    lines=3,
                    show_label=False
                )

                with gr.Row():
                    compare_btn = gr.Button("🔍 Compare", variant="secondary")
                    answer_btn = gr.Button("✨ Generate", variant="primary")

    # SPACING
    gr.HTML("<div style='height:20px'></div>")

    # RETRIEVER
    with gr.Group(elem_classes="glass"):
        gr.HTML("<div class='section-title'>🔎 Retriever Comparison</div>")

        with gr.Row():
            with gr.Column():
                gr.HTML("<span style='color:#60a5fa;font-weight:600'>Similarity Search</span>")
                sim_output = gr.HTML("<div class='scroll'>Waiting for input...</div>")

            with gr.Column():
                gr.HTML("<span style='color:#a78bfa;font-weight:600'>BM25 + MMR Retriever</span>")
                hybrid_output = gr.HTML("<div class='scroll'>Waiting for input...</div>")

    # SPACING
    gr.HTML("<div style='height:20px'></div>")

    # ANSWER SECTION
    with gr.Group(elem_classes="glass"):
        gr.HTML("<div class='section-title'>🧠 Answer Engine</div>")

        with gr.Row():
            with gr.Column():
                gr.HTML("<span style='color:#34d399;font-weight:600'>Top Chunks</span>")
                reranked_output = gr.HTML("<div class='scroll'>...</div>")

            with gr.Column():
                gr.HTML("<span style='color:#fbbf24;font-weight:600'>Final Answer</span>")
                answer_output = gr.Textbox(
                    lines=14,
                    show_label=False,
                    interactive=False,
                    placeholder="Your answer will appear here..."
                )

    # BUTTON LOGIC (unchanged)
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[upload_status]
    )

    compare_btn.click(
        fn=compare_retrievers,
        inputs=[question_input],
        outputs=[sim_output, hybrid_output]
    )

    answer_btn.click(
        fn=generate_answer,
        inputs=[question_input],
        outputs=[reranked_output, answer_output]
    )

    question_input.submit(
        fn=compare_retrievers,
        inputs=[question_input],
        outputs=[sim_output, hybrid_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[upload_status, sim_output, hybrid_output, reranked_output, answer_output]
    )
if __name__ == "__main__":
    app.launch()