import streamlit as st
import os
import uuid
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from graph_backend import get_chatbot_response, embeddings

# 1. Page Configuration
st.set_page_config(page_title="Valve Handbook RAG Chatbot", layout="wide")
st.title("📚 Valve Handbook & Custom PDF Chatbot")
st.write("Upload a PDF to build/rebuild the knowledge base, or chat with the existing database.")

# 2. Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar for PDF Uploads
with st.sidebar:
    st.header("Document Settings")
    uploaded_file = st.file_uploader("Upload a new PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Rebuild Database"):
            with st.spinner("Processing PDF and updating vector store..."):
                # Save uploaded file temporarily
                temp_dir = "data"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Rebuild database logic
                db_path = "./valve_db"
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(pages)
                    
                    # Sanitize null bytes
                    for chunk in chunks:
                        chunk.page_content = chunk.page_content.replace("\x00", "")
                    chunks = [c for c in chunks if c.page_content.strip()]
                    
                    vectorstore = Chroma(embedding_function=embeddings, persist_directory=db_path)
                    
                    # Batch processing
                    for i in range(0, len(chunks), 1):
                        vectorstore.add_documents(chunks[i:i + 1])
                    
                    st.success(f"Successfully processed {len(chunks)} chunks from '{uploaded_file.name}'!")
                    # Clear chat history for the new document context
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid.uuid4())
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Chat Input
if user_query := st.chat_input("Ask a question about your document..."):
    # Display human message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = get_chatbot_response(user_query, thread_id=st.session_state.thread_id)
            st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})