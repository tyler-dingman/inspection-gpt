
import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings

# Load local embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

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

# Helper to split text into chunks
def split_text(text, max_length=500):
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    chunks.append(chunk.strip())
    return chunks

# Initialize Chroma client
def init_chroma():
    return chromadb.Client(Settings(anonymized_telemetry=False))

# Create vector store
chroma_client = init_chroma()
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# Embed and index chunks
def index_chunks(chunks):
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# Search function
def search(query):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return results['documents'][0] if results['documents'] else []

# Streamlit UI
st.title("📄🔍 Local NLP PDF Search Tool (Hosted)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)
    collection.delete()  # Clear previous documents
    index_chunks(chunks)

    st.success("PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about the document:")

    if query:
        results = search(query)
        st.subheader("🔎 Top Answers:")
        for i, res in enumerate(results):
            st.markdown(f"**{i+1}.** {res}")
else:
    st.info("Please upload a PDF to begin.")
