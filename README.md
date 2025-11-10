# 🤖 Bank Statement Analyzer (RAG + OCR → Gemini)

## 📌 Overview

**Bank Statement Analyzer** is a robust Python system designed to extract **structured financial data** and generate **insights** from **unstructured PDF and image bank statements**.  
It implements a **Retrieval-Augmented Generation (RAG)** architecture using **OCR**, **FAISS vector indexing**, and the **Gemini API**.

### 🧩 System Highlights
- **OCR** processes non-selectable text in PDFs and images.  
- **FAISS** builds a high-performance vector database for semantic search.  
- **Gemini** (via `google-genai`) uses a strict system prompt to output a **valid JSON** containing:
  - Account Information  
  - Summary Values  
  - Financial Insights  

Built with **Python**, this solution leverages **Gemini API** and **FAISS** to convert raw, multi-format financial documents into clean, structured, machine-readable records.

---

## 🛠️ Features

### 📄 Multi-Format Ingestion
- Processes **PDFs** (selectable text via PyMuPDF or scanned via OCR).  
- Supports **image formats** such as PNG, JPG, and TIFF.

### 🧠 Intelligent Text Chunking
- Uses **tiktoken** for token-aware segmentation.  
- Creates **overlapping text passages** (default: 800 tokens, 150 overlap) to maintain contextual integrity.

### 🔎 FAISS Vector Indexing
- Uses **Sentence-Transformers** to generate dense embeddings.  
- Stores them in a **disk-backed FAISS index (`rag.index`)** for fast semantic retrieval.

### ✨ Gemini-Powered Extraction
- Leverages **Gemini API** for **zero-shot structured data extraction**.  
- Outputs follow a **strict JSON schema** defined for bank statement analysis.

### 🖥️ Streamlit UI
- Provides an **interactive web interface** for:
  - File upload  
  - Index building/loading/merging  
  - Query input and structured result display  

---

## 🔁 Workflow

1. **Ingest Documents:**  
   Input raw, multi-page PDF or image files.

2. **Perform OCR:**  
   Non-selectable pages are converted to images and processed with **Tesseract OCR**.

3. **Chunk Text:**  
   Split extracted text into **token-limited**, **context-preserving** overlapping passages.

4. **Embed & Index:**  
   Convert chunks into embeddings using **Sentence-Transformers**, then store in **FAISS**.

5. **Query Stage:**  
   Receive user questions via CLI or Streamlit.

6. **Retrieve Context:**  
   Top-K relevant passages are retrieved from the **FAISS** index.

7. **Gemini Generation:**  
   Combine user query + context and send to **Gemini** with a strict JSON output prompt.

8. **Output:**  
   Return a valid structured **JSON** containing all extracted insights.

---

## 🚀 Technologies Used

| Category | Technology |
|-----------|-------------|
| 🐍 Language | Python 3.x |
| ⚙️ LLM API | `google-genai` (Gemini) |
| 🧠 Vector Index | `faiss-cpu`, `sentence-transformers` |
| 🖼️ OCR & PDF | `pytesseract`, `PyMuPDF`, `pdf2image`, `Pillow` |
| 💻 Web UI | `streamlit` |

---

### 💬 Example Query


Extract the account holder name and closing balance.

---

## 🖼️ Sample Extracted JSON

A typical Gemini output follows a **strict schema** and returns a single valid JSON object representing structured financial data.

```json
{
  "account_info": {
    "bank_name": "Example Bank",
    "account_holder_name": "A. B. Customer",
    "masked_account_number": "****4321",
    "statement_month": "2025-10",
    "account_type": "checking"
  },
  "summary_values": {
    "opening_balance": 5000.00,
    "closing_balance": 8500.50,
    "total_credits": 10000.00,
    "total_debits": 6500.50,
    "average_daily_balance": 7000.00,
    "overdraft_count": 0
  },
  "insights": [
    "Account maintained < ₹10,000 average balance during October."
  ]
}
