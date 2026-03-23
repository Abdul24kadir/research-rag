import gradio as gr
import os 
import shutil
import uuid
from dotenv import load_dotenv

from  ingest import ingest_pdf
from retriever import load_vectorstore,build_qa_chain,format_docs


load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"]="false"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Global variables — chain and retriever are built once
# and reused for every question

chain = None
retriever = None


def process_pdf(pdf_file):
    """
    Called when user uploads a PDF.
    Runs the full ingestion pipeline and builds the QA chain.
    Returns a status message shown in the UI.
    """
    global chain, retriever

    if pdf_file is None:
        return "Please upload a PDF FIle first."
    
    try:
        # pdf_file.name gives us the temp file path Gradio created
        print(f"Processing uploaded PDF:{pdf_file.name}")

        #absolute data directory
        data_dir = os.path.join(BASE_DIR,"data")
        os.makedirs(data_dir,exist_ok=True)

        #unique file name
        filename = os.path.basename(pdf_file.name)
        unique_name = f"{uuid.uuid4()}_{filename}"

        #new absolute destination path
        dest_path = os.path.join(data_dir,unique_name)

        #copy temp to project folder
        shutil.copy(pdf_file.name,dest_path)
        print(f"saved file to:{dest_path}")
       

        # Run ingestion — chunks + embeddings + ChromaDB
        ingest_pdf(dest_path)

        # Load vectorstore and build chain
        vectorstore = load_vectorstore()
        chain , retriever = build_qa_chain(vectorstore)

        return f"PDF processed successfully! You can ask Questions."
    
    except Exception as e:
        return f"Error processing PDF:{str(e)}"
    
def answer_question(question):
    """
    Called when user submits a question.
    Returns the answer and formatted citations.
    """
    global chain , retriever
    # Guard — make sure PDF was uploaded first
    if chain is None:
        return "Please upload and process a PDF first.",""
    if not question.strip():
        return "Please enter a question.",""
    
    try:
        #Get answer
        answer = chain.invoke(question)
        # get source citations

        source_docs = retriever.invoke(question)

        citations = ""
        for i,doc in enumerate(source_docs):
            page = doc.metadata.get("page",None)
            page_display = page + 1 if page is not None else "unkown"
            citations +=f"[{i+1}] Page {page_display}:\n"
            citations += f"{doc.page_content[:200]}...\n\n"
        
        return answer ,citations
    except Exception as e:
        return f"Error:{str(e)}",""
    

#Build the Gradio UI
with gr.Blocks(title="Reasearch paper RAG") as app:

    gr.Markdown("#Reasearch Paper Q&A")
    gr.Markdown("Upload a research paper PDF and ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            #Left column - PDF Upload
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Process PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False
            )
        with gr.Column(scale=2):
            #Right column - Q&A
            question_input=gr.Textbox(
                label="Ask a question",
                placeholder="What is the main contribution of this paper?",
                lines = 2
            )
            ask_btn = gr.Button("Ask",variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                lines = 5, 
                interactive=False
            )
            citations_output = gr.Textbox(
                label="Sources",
                lines =8,
                interactive=False
            )
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[upload_status]
    )
    ask_btn.click(
        fn= answer_question,
        inputs=[question_input],
        outputs=[answer_output,citations_output]
    )
    #ALlow pressing enter to submit question
    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output,citations_output]
    )

if __name__=="__main__":
    app.queue().launch(share=True)
    



