# Bank_Account_Analyser
# 🤖 Bank Statement Analyzer (RAG + OCR + Gemini)

## 📌 Overview: Retrieval-Augmented Generation (RAG) System

The **Bank Statement Analyzer** is a powerful Python application designed to extract structured financial data and generate insights from **unstructured PDF and image bank statements**.  

It employs a **Retrieval-Augmented Generation (RAG)** architecture, combining:
- **Optical Character Recognition (OCR)**
- **Advanced text chunking**
- **Vector embeddings (FAISS)**
- **Gemini API**  

This combination enables robust and accurate financial information extraction.  

Built with Python, this solution addresses the challenge of converting raw, multi-format financial documents into clean, machine-readable structured data.

---

## 🛠️ Features

### 📄 Multi-Format Ingestion
- Processes both **PDF documents** (supporting selectable text via **PyMuPDF**) and **non-selectable/scanned documents** via **OCR**.  
- Also supports **image files (PNG, JPG, TIFF)**.

### 🧠 Intelligent Text Chunking
- Uses **tiktoken** to precisely measure token length.  
- Splits pages into overlapping chunks (default **800 tokens**) based on sentence boundaries to preserve context.

### 🔎 FAISS Vector Indexing
- Employs **Sentence-Transformers** and **FAISS** for creating a **high-performance, disk-backed index**.  
- Enables efficient **semantic retrieval** of relevant text passages.

### ✨ Gemini-Powered Extraction
- Leverages the **Gemini API** with a detailed **system prompt** to perform **zero-shot structured data extraction**.  
- Produces a valid **JSON object** containing:
  - Account Info  
  - Summary Values  
  - Financial Insights  

### 🖥️ Streamlit UI
- Provides an **interactive web interface** for:
  - File upload  
  - Index building/loading  
  - Querying and JSON visualization  

---

## ⚙️ Dependencies

The project relies on the following key Python packages (see `requirements.txt`):

- `google-genai` → Gemini API integration  
- `faiss-cpu` → Similarity search & vector indexing  
- `sentence-transformers` → Text embeddings  
- `pytesseract`, `Pillow`, `pdf2image`, `PyMuPDF` → PDF/image parsing & OCR  
- `streamlit` → Interactive web app  

---

## 🔁 Core Workflow: RAG Pipeline

### 1. **Document Ingestion**  
*(Files: `app.py`, `pdf_utils.py`, `ocr_utils.py`)*  
- Extracts selectable text from PDFs via **PyMuPDF**.  
- Converts non-selectable pages to images using **pdf2image** and processes them via **pytesseract** OCR.  

### 2. **Chunking & Indexing**  
*(Files: `chunking.py`, `embeddings_index.py`)*  
- Splits extracted text into **contextual overlapping chunks** using **tiktoken**.  
- Converts chunks to **dense vector embeddings** with **Sentence-Transformers**.  
- Stores embeddings in a **persistent FAISS index (`rag.index`)** linked with `passages.json`.  

### 3. **Interactive Query**  
*(Files: `app.py`, `streamlit_app.py`)*  
- User submits a query (e.g., “Extract all key summary fields”).  
- Query is embedded and compared to FAISS vectors.  
- Retrieves top semantically relevant text passages as **context**.  

### 4. **Generation**  
*(File: `genai_client.py`)*  
- The **retrieved context** and **user query** are sent to the **Gemini model**.  
- The model follows a **strict, bank-specific prompt** to produce a structured **JSON output**.  

---


