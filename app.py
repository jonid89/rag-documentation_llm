import streamlit as st
import os
import uuid
import shutil
from graph_backend import get_chatbot_response, embeddings, vectorstore
from build_db import update_vectorstore_from_pdf

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
                try:
                    num_chunks = update_vectorstore_from_pdf(file_path, vectorstore)
                    
                    st.success(f"Successfully processed {num_chunks} chunks from '{uploaded_file.name}'!")
                    # Clear chat history for the new document context
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid.uuid4())
                    
                    if os.path.exists(file_path):
                        os.remove(file_path)
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