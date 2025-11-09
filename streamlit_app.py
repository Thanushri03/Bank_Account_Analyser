import os
import tempfile
import zipfile
import io
import streamlit as st
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from app import ingest_files, build_index_from_chunks, load_index, FAISS_PATH, PASSAGES_PATH
from genai_client import create_client, ask_gemini
from embeddings_index import EmbeddingsIndex

st.set_page_config(page_title="RAG OCR → Gemini", layout="wide")
st.title("RAG: PDF/Image OCR → FAISS → Gemini")

# -------------------------
# Session state initialization
# -------------------------
if "index_built" not in st.session_state:
    st.session_state["index_built"] = False
if "emb" not in st.session_state:
    st.session_state["emb"] = None
if "num_passages" not in st.session_state:
    st.session_state["num_passages"] = 0

# -------------------------
# API key check
# -------------------------
if not os.getenv("GENAI_API_KEY"):
    st.warning(
        "GENAI_API_KEY not found. Put your key in a .env file or set the GENAI_API_KEY environment variable. "
        "Uploads & index build will be disabled until key is provided."
    )
    upload_disabled = True
else:
    upload_disabled = False

# -------------------------
# File uploaders / controls
# -------------------------
uploaded = st.file_uploader(
    "Upload PDFs or images (multiple)",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    disabled=upload_disabled,
)

build_btn = st.button("Build index from uploaded files", disabled=upload_disabled)

# Uploader for saved index files (.zip or .index + .json)
uploaded_index_files = st.file_uploader(
    "Upload index (.zip) or index + passages files",
    accept_multiple_files=True,
    type=["zip", "index", "json"],
    help="Upload a .zip (containing .index and passages.json) or upload the .index and passages.json files together.",
)

upload_index_btn = st.button("Upload index files")

# Sidebar options
st.sidebar.markdown("## Index file options")
index_filename = st.sidebar.text_input("Index filename", value=FAISS_PATH)
passages_filename = st.sidebar.text_input("Passages filename", value=PASSAGES_PATH)
zip_name = st.sidebar.text_input("Zip filename for download", value="rag_index.zip")
autosave_merged = st.sidebar.checkbox("Autosave merged index", value=False)

# -------------------------
# Build index from uploaded documents (PDF/image)
# -------------------------
if build_btn and uploaded:
    with st.spinner("Saving uploads and building index (this may take a while)..."):
        tmpdir = tempfile.mkdtemp(prefix="rag_upload_")
        paths = []
        for f in uploaded:
            out = os.path.join(tmpdir, f.name)
            with open(out, "wb") as wf:
                wf.write(f.getbuffer())
            paths.append(out)
        chunks_meta = ingest_files(paths)
        if not chunks_meta:
            st.error("No text found in uploads.")
        else:
            emb = build_index_from_chunks(chunks_meta)
            st.session_state["index_built"] = True
            st.session_state["emb"] = emb
            st.session_state["num_passages"] = len(emb.passages)
            st.success(f"Built index with {len(emb.passages)} passages.")

# -------------------------
# Upload index files and MERGE logic (fixed)
# -------------------------
if upload_index_btn:
    if not uploaded_index_files:
        st.error("No files uploaded. Please upload a .zip or both the .index and passages.json files.")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="uploaded_index_")
        saved_paths = []
        try:
            # Save uploaded files to a temp directory
            for uf in uploaded_index_files:
                out_path = os.path.join(tmp_dir, uf.name)
                with open(out_path, "wb") as wf:
                    wf.write(uf.getbuffer())
                saved_paths.append(out_path)

            # Detect whether zip or separate files
            index_path = None
            passages_path = None
            if len(saved_paths) == 1 and saved_paths[0].lower().endswith(".zip"):
                with zipfile.ZipFile(saved_paths[0], "r") as zf:
                    zf.extractall(tmp_dir)
                for fn in os.listdir(tmp_dir):
                    if fn.lower().endswith(".index"):
                        index_path = os.path.join(tmp_dir, fn)
                    if fn.lower().endswith(".json"):
                        passages_path = os.path.join(tmp_dir, fn)
            else:
                for p in saved_paths:
                    if p.lower().endswith(".index"):
                        index_path = p
                    if p.lower().endswith(".json"):
                        passages_path = p

            if not index_path or not passages_path:
                st.error("Could not find both index (.index) and passages (.json) files in the upload. Please upload a zip with both or upload both files.")
            else:
                # Load uploaded index into temporary EmbeddingsIndex
                emb_uploaded = EmbeddingsIndex()
                try:
                    emb_uploaded.load(index_path=index_path, passages_path=passages_path)
                except Exception as e_load:
                    st.error(f"Failed to load uploaded index: {e_load}")
                    raise
                # Debug: uploaded count
                uploaded_count = len(getattr(emb_uploaded, "passages", []) or [])
                st.info(f"Uploaded index contains {uploaded_count} passages.")

                # Check existing in-memory index
                emb_existing = st.session_state.get("emb")
                existing_count = len(getattr(emb_existing, "passages", []) or [])
                st.info(f"Existing in-memory index has {existing_count} passages (0 if not built).")

                if emb_existing is None or existing_count == 0:
                    # No existing index in memory (or empty) -> use uploaded as current
                    st.info("No existing in-memory index found (or it is empty). Using uploaded index as current index.")
                    st.session_state["emb"] = emb_uploaded
                    st.session_state["index_built"] = True
                    st.session_state["num_passages"] = uploaded_count
                    st.success(f"Uploaded index loaded with {uploaded_count} passages.")
                    if autosave_merged:
                        try:
                            emb_uploaded.save(index_path=index_filename, passages_path=passages_filename)
                            st.info(f"Autosaved uploaded index to {index_filename} and {passages_filename}.")
                        except Exception as e_save:
                            st.error(f"Autosave failed: {e_save}")
                else:
                    # Merge: combine passages (existing first), dedupe exact matches, rebuild
                    st.info("Merging uploaded index with existing in-memory index. This will re-build the combined FAISS index (may take time).")

                    combined_passages = list(emb_existing.passages) + list(emb_uploaded.passages)

                    # Simple exact-text dedupe preserving order (existing-first)
                    seen = set()
                    deduped_passages = []
                    for p in combined_passages:
                        if p in seen:
                            continue
                        seen.add(p)
                        deduped_passages.append(p)

                    st.info(f"Combined {existing_count} + {uploaded_count} => {len(deduped_passages)} after deduplication.")

                    # Rebuild embeddings + FAISS from combined deduped passages
                    new_emb = EmbeddingsIndex(model_name=emb_existing.model_name)
                    try:
                        with st.spinner("Rebuilding combined index (encoding embeddings)..."):
                            new_emb.build(deduped_passages)
                    except Exception as e_build:
                        st.error(f"Failed to build merged index: {e_build}")
                    else:
                        # Replace in-memory index with merged one
                        st.session_state["emb"] = new_emb
                        st.session_state["index_built"] = True
                        st.session_state["num_passages"] = len(new_emb.passages)
                        st.success(f"Merged index loaded with {len(new_emb.passages)} passages (deduplicated).")

                        # Autosave if requested
                        if autosave_merged:
                            try:
                                new_emb.save(index_path=index_filename, passages_path=passages_filename)
                                st.info(f"Autosaved merged index to {index_filename} and {passages_filename}.")
                            except Exception as e_save:
                                st.error(f"Autosave failed: {e_save}")

        except Exception as e:
            st.error(f"Upload/load failed: {e}")

# -------------------------
# If index available — show UI for save/download/query
# -------------------------
if st.session_state["index_built"] and st.session_state["emb"] is not None:
    num_pass = st.session_state.get("num_passages", 0)
    st.markdown(f"**Index loaded with {num_pass} passages.**")

    # Save index button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save index to disk"):
            try:
                emb: EmbeddingsIndex = st.session_state["emb"]
                emb.save(index_path=index_filename, passages_path=passages_filename)
                st.success(f"Index saved to '{index_filename}' and passages saved to '{passages_filename}'.")
            except Exception as e:
                st.error(f"Failed to save index: {e}")

    # Download index as zip
    with col2:
        if st.button("Create download ZIP"):
            try:
                emb: EmbeddingsIndex = st.session_state["emb"]
                save_index_path = index_filename
                save_passages_path = passages_filename
                try:
                    emb.save(index_path=save_index_path, passages_path=save_passages_path)
                except Exception:
                    tmp_dir = tempfile.mkdtemp(prefix="rag_index_")
                    save_index_path = os.path.join(tmp_dir, os.path.basename(index_filename))
                    save_passages_path = os.path.join(tmp_dir, os.path.basename(passages_filename))
                    emb.save(index_path=save_index_path, passages_path=save_passages_path)

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(save_index_path, arcname=os.path.basename(save_index_path))
                    zipf.write(save_passages_path, arcname=os.path.basename(save_passages_path))
                zip_buffer.seek(0)

                st.download_button(
                    label="Download index (.zip)",
                    data=zip_buffer,
                    file_name=zip_name,
                    mime="application/zip",
                )
                st.success("ZIP created. Click the download button above.")
            except Exception as e:
                st.error(f"Failed to create/download ZIP: {e}")

    st.markdown("---")
    st.markdown("### Ask a question (uses the built/loaded FAISS index)")
    question = st.text_input("Question")
    top_k = 8
    if st.button("Ask Gemini") and question.strip():
        try:
            client = create_client()
        except Exception as e:
            st.error(f"GENAI client creation failed: {e}")
            client = None

        if client:
            emb: EmbeddingsIndex = st.session_state["emb"]
            retrieved = emb.query(question, top_k=top_k)
            if not retrieved:
                st.info("No relevant passages found.")
            else:
                
                top_texts = [r["passage"] for r in retrieved]
                with st.spinner("Asking Gemini..."):
                    answer = ask_gemini(client, question, top_texts)
                st.write("### Gemini answer")
                st.write(answer)
else:
    st.info("Build an index from uploads or load an existing index to start querying.")
