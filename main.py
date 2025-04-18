import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np

# Load local embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Helper to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with fitz.open(tmp_path) as doc:
        for page in doc:
            text += page.get_text()
    os.remove(tmp_path)
    return text

# Streamlit UI
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Flag_of_Iowa.svg/1200px-Flag_of_Iowa.svg.png", width=180)
with col2:
    st.title("📄🔍 Iowa Fire Inspection Chat GPT Tool")

uploaded_files = st.file_uploader("Upload one or more PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    splitter = CharacterTextSplitter(separator=". ", chunk_size=500, chunk_overlap=50)

    for uploaded_file in uploaded_files:
        raw_text = extract_text_from_pdf(uploaded_file)
        texts = splitter.split_text(raw_text)
        all_docs.extend([Document(page_content=t) for t in texts])

    # Create FAISS index with all documents
    db = FAISS.from_documents(all_docs, embeddings)
    st.success("PDFs indexed. You can now ask questions across all files.")

    query = st.text_input("Ask a question about the uploaded documents:", key="query", placeholder="Type your question here...", help="This searches across all uploaded PDFs.", label_visibility="visible")

    if query:
        results = db.similarity_search(query, k=5)
        st.subheader("🔎 Top Answers:")
        for i, res in enumerate(results):
            st.markdown(f"**{i+1}.** {res.page_content}")
else:
    st.info("Please upload one or more PDF documents to begin.")
