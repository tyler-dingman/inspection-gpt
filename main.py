
import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np

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

# Embed and index chunks
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# Search function
def search(query, index, chunks):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = [chunks[i] for i in I[0]]
    return results

# Streamlit UI
st.title("📄🔍 Local NLP PDF Search Tool")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)
    index, _, chunk_list = create_faiss_index(chunks)

    st.success("PDF indexed. You can now ask questions.")

    query = st.text_input("Ask a question about the document:")

    if query:
        results = search(query, index, chunk_list)
        st.subheader("🔎 Top Answers:")
        for i, res in enumerate(results):
            st.markdown(f"**{i+1}.** {res}")
else:
    st.info("Please upload a PDF to begin.")
