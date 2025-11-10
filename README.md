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

## ⚙️ Pseudo Code: Ingestion & Indexing

The following pseudo code outlines the **core ingestion and indexing pipeline** for the Bank Statement Analyzer.  
It demonstrates how PDFs and images are processed, chunked, embedded, and stored in the FAISS vector index.

---

### 🧩 Ingestion Function

```pseudo
FUNCTION ingest_files(PATHS):
  CHUNKS_META = []
  FOR each PATH in PATHS:
    NAME = basename(PATH)
    IF PATH is PDF:
      PAGES = pdf_pages_text(PATH)  // Tries selectable text first
      FOR each PAGE_NO, TEXT in PAGES:
        IF TEXT is empty:
          // Fallback to OCR for scanned pages
          IMAGE = pdf_to_images(PATH)[PAGE_NO - 1]
          TEXT = image_to_text(IMAGE)
        END IF
        // Chunk text into passages with source/page metadata
        META = chunk_page_text(TEXT, SOURCE=NAME, PAGE_NO=PAGE_NO)
        CHUNKS_META.EXTEND(META)
    ELSE IF PATH is Image:
      IMAGE = open_image(PATH)
      TEXT = image_to_text(IMAGE)
      META = chunk_page_text(TEXT, SOURCE=NAME, PAGE_NO=1)
      CHUNKS_META.EXTEND(META)
    END IF
  RETURN CHUNKS_META
END FUNCTION
```

---

### 🧠 Index Building Function

```pseudo

FUNCTION build_index_from_chunks(CHUNKS_META):
  PASSAGES = [c["label"] for c in CHUNKS_META]
  EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
  VECTORS = EMBEDDER.encode(PASSAGES)
  Normalize_L2(VECTORS)
  FAISS_INDEX = IndexFlatIP(DIM)
  FAISS_INDEX.add(VECTORS)
  FAISS_INDEX.save("rag.index")
  PASSAGES.save("passages.json")
  RETURN EmbeddingsIndex
END FUNCTION

```

### 🧠 Query Handling & Retrieval Function

```pseudo
FUNCTION query_rag_system(QUERY):
  // Load FAISS index and passage metadata
  FAISS_INDEX = load_faiss_index("rag.index")
  PASSAGES = load_passages("passages.json")

  // Embed the user query
  EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
  QUERY_VECTOR = EMBEDDER.encode([QUERY])
  Normalize_L2(QUERY_VECTOR)

  // Retrieve top-k relevant passages
  TOP_K = 5
  SCORES, IDS = FAISS_INDEX.search(QUERY_VECTOR, TOP_K)
  CONTEXT = [PASSAGES[i] for i in IDS]

  // Combine query and retrieved context
  PROMPT = format_prompt(QUERY, CONTEXT)

  // Call Gemini API with strict JSON system prompt
  RESPONSE = GEMINI.generate(
                model="gemini-pro",
                system_prompt="Return structured JSON for bank statement data.",
                user_prompt=PROMPT
             )

  // Parse and return JSON output
  RESULT = parse_json(RESPONSE)
  RETURN RESULT
END FUNCTION

```


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
