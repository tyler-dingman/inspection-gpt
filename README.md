# 📄🔍 Local NLP PDF Search Tool

This is a simple web app that lets you upload a PDF and ask natural language questions about its contents — all **without using the internet or any API keys**.

## 🚀 Features
- 📤 Upload PDFs
- 🔎 Ask questions in plain English
- 🧠 Powered by local embeddings with `sentence-transformers`
- 💬 Simple Streamlit interface
- 🛠 Runs 100% locally — no OpenAI key needed

## 🛠 Setup Instructions

1. **Clone the repo:**
```bash
git clone https://github.com/tyler-dingman/inspection-gpt.git
cd inspection-gpt
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run main.py
```

4. **Use the web interface:**
- Upload a PDF
- Ask your questions
- View top matching answers

## 📦 Requirements
- Python 3.8+

## ✨ Credits
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

## 🔐 Privacy
This tool runs locally on your machine and never sends any data to external servers.