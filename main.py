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
st.title("📄🔍 Local NLP PDF Search Tool (FAISS + LangChain)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Step 1: Extract text
    raw_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Split into chunks
    splitter = CharacterTextSplitter(separator=". ", chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]

    # Step 3: Create FAISS index
    db = FAISS.from_documents(docs, embeddings)
    st.success("PDF indexed. You can now ask questions.")

    # Step 4: Accept questions
    query = st.text_input("Ask a question about the document:")

    if query:
        results = db.similarity_search(query, k=5)
        st.subheader("🔎 Top Answers:")
        for i, res in enumerate(results):
            st.markdown(f"**{i+1}.** {res.page_content}")
else:
    st.info("Please upload a PDF to begin.")
