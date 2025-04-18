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
    "https://dial.iowa.gov/sites/default/files/documents/2023/05/plan_of_
