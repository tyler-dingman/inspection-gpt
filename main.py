import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import requests
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

# PDF links from Iowa DIAL page
PDF_URLS = [
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/abhr_guidance.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/fire_door_inspection_form.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/electrical_receptacle_testing_form.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/emergency_lighting_test_log.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/fire_watch_plan_guidance.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/health_care_door_locking_guide.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/evacuation_plan_guide.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/surge_protector_policy.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/holiday_decorating_memo.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/plan_of_correction_template.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/sensitivity_testing_enforcement.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/sprinkler_system_outage_requirements.pdf",
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/waiver_form.pdf"
]

# Function to download and extract PDF text
def download_and_extract_pdfs(pdf_urls):
    all_docs = []
    splitter = CharacterTextSplitter(separator=". ", chunk_size=500, chunk_overlap=50)

    for url in pdf_urls:
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp.seek(0)
            tmp_path = tmp.name

            text = ""
            try:
                doc = fitz.open(stream=tmp.read(), filetype="pdf")
                for page in doc:
                    text += page.get_text()
                doc.close()
            except Exception as e:
                st.warning(f"Skipping {url} due to error: {str(e)}")
                os.remove(tmp_path)
                continue

        os.remove(tmp_path)
        chunks = splitter.split_text(text)
        all_docs.extend([Document(page_content=chunk) for chunk in chunks])

    return all_docs

# Streamlit UI
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Flag_of_Iowa.svg/1200px-Flag_of_Iowa.svg.png", width=180)
with col2:
    st.title("📄🔍 Iowa Fire Safety Docs Search Tool")

if st.button("Build and Index Text from Iowa DIAL PDFs"):
    st.info("Downloading and processing PDFs...")
    all_docs = download_and_extract_pdfs(PDF_URLS)
    if not all_docs:
        st.error("No documents were processed successfully. Please check the PDF sources.")
    else:
        db = FAISS.from_documents(all_docs, embeddings)
        st.success("All documents indexed. You can now search them below.")

        query = st.text_input(
            "Ask a question about the documents:",
            key="query",
            placeholder="Type your question here...",
            help="This searches across all the Iowa DIAL PDF documents.",
            label_visibility="visible"
        )

        if query:
            results = db.similarity_search(query, k=5)
            st.subheader("🔎 Top Answers:")
            for i, res in enumerate(results):
                st.markdown(f"**{i+1}.** {res.page_content}")
else:
    st.info("Click the button above to fetch and search PDFs from the Iowa DIAL Fire Safety page.")
