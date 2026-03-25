# app.py (root level — for HuggingFace Spaces deployment)

import sys
import os

# add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import shutil
import uuid
from dotenv import load_dotenv
from ingest import ingest_pdf
from retriever import load_vectorstore, build_qa_chain

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# global state
chain = None
retriever = None
vectorstore_global = None


def reset_chain():
    global chain, retriever, vectorstore_global
    chain = None
    retriever = None
    if vectorstore_global is not None:
        try:
            vectorstore_global._client._system.stop()
        except:
            pass
        vectorstore_global = None
    import gc
    gc.collect()
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
        except:
            pass


def process_pdf(pdf_file):
    global chain, retriever, vectorstore_global

    if pdf_file is None:
        return "Please upload a PDF file first."

    try:
        reset_chain()

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = os.path.basename(pdf_file.name)
        unique_name = f"{uuid.uuid4()}_{filename}"
        dest_path = os.path.join(data_dir, unique_name)
        shutil.copy(pdf_file.name, dest_path)

        ingest_pdf(dest_path)

        vectorstore_global = load_vectorstore()
        chain, retriever = build_qa_chain(vectorstore_global)

        return "PDF processed successfully! You can now ask questions."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def answer_question(question):
    global chain, retriever

    if chain is None:
        return "Please upload and process a PDF first.", ""

    if not question.strip():
        return "Please enter a question.", ""

    try:
        answer = chain.invoke(question)

        source_docs = retriever.invoke(question)
        citations = ""
        for i, doc in enumerate(source_docs):
            page = doc.metadata.get("page", None)
            page_display = page + 1 if page is not None else "unknown"
            citations += f"[{i+1}] Page {page_display}:\n"
            citations += f"{doc.page_content[:200]}...\n\n"

        return answer, citations

    except Exception as e:
        return f"Error: {str(e)}", ""


def clear_all():
    reset_chain()
    return "Cleared! Upload a new PDF to continue.", "", ""


# build UI
with gr.Blocks(title="Research Paper Q&A") as app:

    gr.Markdown("# Research Paper Q&A")
    gr.Markdown("Upload a research paper PDF and ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Process PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False
            )
            clear_btn = gr.Button("Clear & load new PDF", variant="secondary")

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="What is the main contribution of this paper?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                lines=5,
                interactive=False
            )
            citations_output = gr.Textbox(
                label="Sources",
                lines=8,
                interactive=False
            )

    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[upload_status]
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, citations_output]
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, citations_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[upload_status, answer_output, citations_output]
    )


if __name__ == "__main__":
    app.launch()