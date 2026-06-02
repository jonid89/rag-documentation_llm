import streamlit as st
import os
import uuid
import shutil
import tempfile
from src.graph_backend import get_chatbot_response, embeddings, vectorstore, MODEL_REGISTRY
from src.build_db import update_vectorstore_from_pdf, get_user_temp_db_path, cleanup_old_temp_dbs
from langchain_chroma import Chroma

# 1. Page Configuration
st.set_page_config(page_title="Valve Handbook RAG Chatbot", layout="wide")
st.title("💬 Ask the Chatbot")
st.write("Ask questions about your document. You can upload a PDF to build/rebuild the knowledge base, or chat with the existing database.")

# 1.5. Startup Cleanup
# Clean up old temporary databases on app startup
cleanup_old_temp_dbs(hours=2)

# 2. Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_db_dir" not in st.session_state:
    st.session_state.user_db_dir = ""  # Empty string means use default master DB

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Gemini 3.1 Flash Lite"  # Default to Gemini 3.1 Flash Lite

if "current_document" not in st.session_state:
    st.session_state.current_document = "Valve Handbook (Default)"  # Default document name

# 3. Sidebar for PDF Uploads and Model Selection
with st.sidebar:
    st.header("Settings")
    
    # Document Upload Section
    st.subheader("📄 Upload Document")
    
    # Show current document
    st.caption(f"Currently using: **{st.session_state.current_document}**")
    
    uploaded_file = st.file_uploader("Upload a new PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Rebuild Database"):
            with st.spinner("Processing PDF and updating vector store..."):
                # Clean up before creating new user DB
                cleanup_old_temp_dbs(hours=2)
                
                # Get user-specific temp database path
                user_db_dir = get_user_temp_db_path(st.session_state.thread_id)
                os.makedirs(user_db_dir, exist_ok=True)
                
                # Save uploaded file temporarily
                # Determine writable temp directory
                if not os.access(".", os.W_OK):
                    temp_dir = os.path.join(tempfile.gettempdir(), "rag_uploads")
                else:
                    temp_dir = "data"
                
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Rebuild user-specific database
                try:
                    # Create a fresh vectorstore for this user's temp DB
                    from langchain_chroma import Chroma
                    user_vectorstore = Chroma(
                        persist_directory=user_db_dir,
                        embedding_function=embeddings
                    )
                    
                    num_chunks = update_vectorstore_from_pdf(file_path, user_vectorstore)
                    
                    # Store the user's DB path and document name in session state
                    st.session_state.user_db_dir = user_db_dir
                    st.session_state.current_document = uploaded_file.name
                    
                    st.success(f"Successfully processed {num_chunks} chunks from '{uploaded_file.name}'!")
                    # Clear chat history for the new document context
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid.uuid4())
                    
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Clear user_db_dir on error
                    st.session_state.user_db_dir = ""
    
    st.divider()
    
    # Model selection dropdown
    st.subheader("🤖 LLM Model")
    st.session_state.selected_model = st.selectbox(
        "Choose a model:",
        options=list(MODEL_REGISTRY.keys()),
        index=list(MODEL_REGISTRY.keys()).index(st.session_state.selected_model),
        help="Select which Gemini model to use for responses"
    )

    st.divider()
    
    # Portfolio / GitHub Links
    st.subheader("🔗 Links")
    st.markdown("[⭐ View Source on GitHub](https://github.com/jonid89/rag-documentation_llm)")

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
            ai_response = get_chatbot_response(
                user_query, 
                thread_id=st.session_state.thread_id,
                db_dir=st.session_state.user_db_dir,
                model=st.session_state.selected_model
            )
            st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})