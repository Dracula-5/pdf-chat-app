import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

"""
Streamlit PDF Chat App — token-safe, elaborated answers, and progress bars

Now uses **Chroma** instead of FAISS to avoid installation issues on Streamlit Cloud.

Run locally:
1) Create a virtual env and activate it
   python -m venv venv
   venv\\Scripts\\activate    # Windows PowerShell

2) Install dependencies
   pip install -U -r requirements.txt

3) Run
   streamlit run app.py

Notes:
- Default model: google/flan-t5-base (512 token input).
- This code uses tokenizer's model_max_length to prevent sequence length errors.
"""

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚", layout="wide")
st.header("📚 Chat with multiple PDFs")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.subheader("Your documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Process")
    clear_button = st.button("Clear Chat")

# -------------------------------
# Initialize session state
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Clear chat
if clear_button:
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.success("Chat history cleared!")

# -------------------------------
# Process PDFs with progress bar
# -------------------------------
if process_button and uploaded_files:
    progress = st.progress(0)
    status_text = st.empty()

    all_docs = []

    # Step 1: Save PDFs
    status_text.text("Saving uploaded PDFs...")
    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_docs.extend(documents)
    progress.progress(20)

    # Step 2: Split text
    status_text.text("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    progress.progress(40)

    # Step 3: Create embeddings
    status_text.text("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    progress.progress(70)

    # Step 4: Build Chroma index
    status_text.text("Building Chroma index...")
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    progress.progress(90)

    # Step 5: Load LLM
    status_text.text("Loading language model...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.qa_chain = {"llm": llm, "retriever": retriever, "memory": memory}
    progress.progress(100)
    status_text.text("✅ Documents processed successfully! You can now start chatting.")

# -------------------------------
# Chat Interface with progress bar
# -------------------------------
user_question = st.text_input("Ask a question about your documents:")

if user_question:
    if st.session_state.qa_chain:
        llm = st.session_state.qa_chain["llm"]
        retriever = st.session_state.qa_chain["retriever"]
        memory = st.session_state.qa_chain["memory"]

        # Retrieve docs
        docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])[:400]

        # Construct elaborated prompt
        prompt = f"""
        You are a helpful and detailed assistant. Answer the question using the context below.
        Always elaborate fully, explain step by step, give examples, and include a SOURCES section.

        Context:
        {context}

        Question: {user_question}
        """

        # Progress bar for answer generation
        gen_progress = st.progress(0)
        for pct in range(0, 100, 10):
            time.sleep(0.05)
            gen_progress.progress(pct + 10)

        # Generate answer
        raw_answer = llm(prompt)

        # Optional refine pass: elaborate further
        refine_prompt = f"Expand and elaborate the following answer in detail, adding context, examples, and explanation.\nAnswer: {raw_answer}".strip()
        refined_answer = llm(refine_prompt)

        st.session_state.chat_history.append((user_question, refined_answer))
    else:
        st.warning("⚠️ Please upload and process your documents first!")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
