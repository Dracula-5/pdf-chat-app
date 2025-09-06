"""
Streamlit PDF Chat App — token-safe, elaborated answers, and progress bars

This version fixes the "token indices sequence length" error by truncating the retrieved
context so the full prompt (instructions + context + question) fits within the model's
maximum input length. It also keeps the elaboration/refine behavior and shows progress
bars during PDF indexing and answer generation.

Run locally:
1) Create a virtual env and activate it
   python -m venv venv
   venv\\Scripts\\activate    # Windows PowerShell

2) Install dependencies
   pip install -U streamlit transformers sentence-transformers langchain langchain-community langchain-huggingface torch faiss-cpu

3) Run
   streamlit run app.py

Notes:
- Default model: google/flan-t5-base (512 token input). If you have a GPU and more memory,
  you may switch to a model that supports larger inputs.
- This code uses the tokenizer's model_max_length to determine the truncation threshold.
"""

import streamlit as st
import tempfile
import os
import time
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# Small UI templates
# -------------------------------
css = """
<style>
.chat-message { padding: 0.8rem; border-radius: 0.6rem; margin-bottom: 0.8rem; display:flex; }
.chat-message.user { background:#2b313e; color: #fff; }
.chat-message.bot { background:#475063; color: #fff; }
.chat-message .message { padding-left: 0.8rem; white-space: pre-wrap; }
</style>
"""

st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚", layout="wide")
st.markdown(css, unsafe_allow_html=True)
st.header("📚 Chat with multiple PDFs — Token-safe & Elaborated Answers")

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.subheader("Configuration")
    model_name = st.selectbox("LLM model (local)", options=["google/flan-t5-small", "google/flan-t5-base"], index=1)
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.number_input("Chunk size (chars)", value=500, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=100, step=50)
    top_k = st.slider("Retriever top_k", min_value=1, max_value=8, value=3)
    reserve_output = st.slider("Reserved output tokens", min_value=64, max_value=256, value=128)
    refine_toggle = st.checkbox("Enable refine pass (slower, more detailed)", value=True)
    show_sources = st.checkbox("Show source excerpts", value=True)
    st.markdown("---")
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Process (index)")
    clear_button = st.button("Clear chat & index")

# -------------------------------
# Session state
# -------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Helpers for tokenizer/pipeline
# -------------------------------
@st.cache_resource
def load_tokenizer_and_pipeline(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device_id = 0 if device == "cuda" else -1
    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        # do not set temperature here; control sampling at call-time if needed
    )
    return tokenizer, pipe

@st.cache_resource
def init_embeddings(embedding_model_name: str, device: str):
    return HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

@st.cache_data
def load_and_split(files, chunk_size: int, chunk_overlap: int) -> Tuple[Tuple[str, dict], ...]:
    all_splits = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp.flush()
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splits = splitter.split_documents(docs)
            for s in splits:
                meta = s.metadata.copy() if hasattr(s, "metadata") else {}
                meta["source"] = getattr(f, "name", "uploaded.pdf")
                all_splits.append((s.page_content, meta))
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return tuple(all_splits)

# Build prompt ensuring token-length safety
def build_prompt_with_truncation(tokenizer, instruction: str, question: str, docs: List[Document],
                                 max_input_tokens: int, reserve_output_tokens: int) -> Tuple[str, List[Tuple[int, str]]]:
    """Return prompt (string) that fits within max_input_tokens when tokenized.
    Also return list of used docs as (index, excerpt).
    """
    prefix = instruction + "\n\nContext:\n"
    suffix = f"\n\nQuestion: {question}\nAnswer:\n"

    # Count tokens for prefix and suffix
    try:
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
    except Exception:
        # fallback if tokenizer.encode signature differs
        prefix_tokens = len(tokenizer(prefix)['input_ids'])
        suffix_tokens = len(tokenizer(suffix)['input_ids'])

    available = max_input_tokens - reserve_output_tokens - prefix_tokens - suffix_tokens
    if available < 0:
        # not enough room even without context; reduce reserve_output
        available = max_input_tokens - prefix_tokens - suffix_tokens
        if available < 0:
            # last resort: truncate the question (shouldn't happen normally)
            question = question[:200]
            suffix = f"\n\nQuestion: {question}\nAnswer:\n"
            try:
                suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
            except Exception:
                suffix_tokens = len(tokenizer(suffix)['input_ids'])
            available = max_input_tokens - reserve_output_tokens - prefix_tokens - suffix_tokens
            if available < 0:
                available = 0

    used = []
    accumulated_tokens = 0
    pieces = []

    for idx, d in enumerate(docs):
        text = d.page_content.replace('\n', ' ').strip()
        if not text:
            continue
        # tokenize text
        try:
            toks = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            toks = tokenizer(text)['input_ids']
        tlen = len(toks)
        if accumulated_tokens + tlen <= available:
            pieces.append(text)
            accumulated_tokens += tlen
            used.append((idx, text[:400]))
        else:
            # take a slice of tokens that fits
            remain = available - accumulated_tokens
            if remain <= 0:
                break
            excerpt = tokenizer.decode(toks[:remain], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pieces.append(excerpt + '...')
            used.append((idx, excerpt[:400]))
            accumulated_tokens += remain
            break

    context_text = "\n\n".join(pieces)
    prompt = prefix + context_text + suffix
    return prompt, used

# -------------------------------
# Processing (indexing) with progress bar
# -------------------------------
if process_button and uploaded_files:
    progress = st.progress(0)
    status = st.empty()

    status.text("Saving uploaded PDFs...")
    progress.progress(5)

    splits = load_and_split(uploaded_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    progress.progress(25)
    status.text("Creating embeddings...")

    embeddings = init_embeddings(embedding_model, device)
    progress.progress(45)
    status.text("Building FAISS index...")

    docs = [Document(page_content=text, metadata=meta) for (text, meta) in splits]
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorstore = vectorstore
    progress.progress(70)

    status.text("Loading tokenizer and model (this may take a while)...")
    tokenizer, pipe = load_tokenizer_and_pipeline(model_name, device)
    st.session_state.tokenizer = tokenizer
    st.session_state.pipe = pipe
    progress.progress(100)
    status.text("✅ Index and model ready. You can ask questions now.")

# Clear
if clear_button:
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.session_state.tokenizer = None
    st.session_state.pipe = None
    st.success("Cleared chat and index.")

# -------------------------------
# Chat interface
# -------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_input("Ask a question about your documents:")
    if st.button("Send") and user_question:
        if st.session_state.vectorstore is None or st.session_state.pipe is None:
            st.warning("Please upload, process your documents, and wait for the model to load.")
        else:
            # retrieve
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            candidate_docs = retriever.get_relevant_documents(user_question)

            tokenizer = st.session_state.tokenizer
            pipe = st.session_state.pipe
            max_input = getattr(tokenizer, 'model_max_length', 512)
            # ensure a reasonable cap
            if max_input is None or max_input <= 0:
                max_input = 512

            instruction = (
                "You are an expert assistant. Answer thoroughly and step-by-step. Provide examples, clarifications, "
                "and a final SOURCES section listing which document excerpts you used. Be explicit and educational."
            )

            prompt, used_docs = build_prompt_with_truncation(
                tokenizer=tokenizer,
                instruction=instruction,
                question=user_question,
                docs=candidate_docs,
                max_input_tokens=max_input,
                reserve_output_tokens=reserve_output,
            )

            # show progress bar for generation
            gen_progress = st.progress(0)
            gen_status = st.empty()
            gen_status.text("Generating answer — preparing...")
            gen_progress.progress(10)

            gen_status.text("Generating answer — running model...")
            gen_progress.progress(40)
            # generate (use max_new_tokens to control output size)
            try:
                outputs = pipe(prompt, max_new_tokens=reserve_output)
            except TypeError:
                # older transformers versions may not accept max_new_tokens in pipeline call
                outputs = pipe(prompt, max_length=reserve_output)

            raw = outputs[0]['generated_text'] if isinstance(outputs, list) and isinstance(outputs[0], dict) else str(outputs)
            gen_progress.progress(70)

            # optional refine pass
            final_answer = raw
            if refine_toggle:
                gen_status.text("Refining and elaborating answer...")
                refine_instruction = (
                    "Improve and expand the previous answer. Add more examples, explanations, and clarify assumptions. "
                    "Keep the same SOURCES. If nothing new, keep the answer unchanged."
                )
                refine_prompt = f"{refine_instruction}\n\nPrevious answer:\n{raw}\n\nContext (if useful):\n"
                # reuse the same used docs, but ensure the refine prompt also fits
                refine_docs = [candidate_docs[i] for i, _ in used_docs] if used_docs else []
                refine_prompt_text, _ = build_prompt_with_truncation(
                    tokenizer=tokenizer,
                    instruction=refine_instruction + "\n\nContext:",
                    question=user_question,
                    docs=refine_docs,
                    max_input_tokens=max_input,
                    reserve_output_tokens=reserve_output,
                )
                try:
                    out2 = pipe(refine_prompt_text + "\n\nPlease produce the refined answer:", max_new_tokens=reserve_output)
                except TypeError:
                    out2 = pipe(refine_prompt_text + "\n\nPlease produce the refined answer:", max_length=reserve_output)
                raw2 = out2[0]['generated_text'] if isinstance(out2, list) and isinstance(out2[0], dict) else str(out2)
                final_answer = raw2
                gen_progress.progress(95)

            gen_status.text("Done.")
            gen_progress.progress(100)

            # store and display
            st.session_state.chat_history.append((user_question, final_answer, used_docs))

# Display chat history
with col1:
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, a, used in reversed(st.session_state.chat_history[-20:]):
            st.markdown(f"<div class='chat-message user'><div class='message'><b>You:</b> {q}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message bot'><div class='message'><b>Assistant:</b><br>{a.replace('\n','<br>')}</div></div>", unsafe_allow_html=True)
            if show_sources and used:
                with st.expander("Show used excerpts"):
                    for idx, excerpt in used:
                        md = candidate_docs[idx].metadata if idx < len(candidate_docs) else {}
                        src = md.get('source','unknown')
                        page = md.get('page','')
                        st.markdown(f"**[DOC_{idx}]** {src} (page: {page})")
                        st.code(excerpt)
                        st.markdown("---")

with col2:
    st.markdown("### Status & Controls")
    st.write("Model:", model_name)
    st.write("Embeddings:", embedding_model)
    st.write("Retriever top_k:", top_k)
    st.write("Reserved output tokens:", reserve_output)
    st.markdown("---")

# -------------------------------
# End
# -------------------------------

# If you run into any runtime error, paste the traceback here and I will update the code further.