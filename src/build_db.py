import os
import shutil
import streamlit as st
import tempfile
import time
from datetime import datetime, timedelta
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def get_user_temp_db_path(thread_id):
    """
    Returns the path to a user-specific temporary database folder.
    
    Args:
        thread_id: Unique identifier for the user session
        
    Returns:
        Absolute path to the temp database directory
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Fallback to system temp if project dir is read-only
    if not os.access(root_dir, os.W_OK):
        base_temp_dir = os.path.join(tempfile.gettempdir(), "temp_dbs")
    else:
        base_temp_dir = os.path.join(root_dir, "temp_dbs")
    
    return os.path.join(base_temp_dir, thread_id)

def cleanup_old_temp_dbs(hours=4):
    """
    Removes temporary database folders older than specified hours.
    This is a passive cleanup that runs on app startup and before new uploads.
    
    Args:
        hours: Number of hours after which a temp DB is considered stale (default: 4)
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine temp_dbs location
    if not os.access(root_dir, os.W_OK):
        temp_dbs_dir = os.path.join(tempfile.gettempdir(), "temp_dbs")
    else:
        temp_dbs_dir = os.path.join(root_dir, "temp_dbs")
    
    # Skip if temp_dbs directory doesn't exist yet
    if not os.path.exists(temp_dbs_dir):
        return
    
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    
    try:
        for folder_name in os.listdir(temp_dbs_dir):
            folder_path = os.path.join(temp_dbs_dir, folder_name)
            
            if os.path.isdir(folder_path):
                # Get folder modification time
                folder_mtime = os.path.getmtime(folder_path)
                
                # Delete if older than cutoff
                if folder_mtime < cutoff_time:
                    shutil.rmtree(folder_path)
                    print(f"Cleaned up old temp database: {folder_path}")
    except Exception as e:
        print(f"Warning: Error during temp database cleanup: {e}")

def update_vectorstore_from_pdf(file_path, vectorstore):
    """
    Loads a PDF, splits it into chunks, sanitizes content, 
    clears existing documents, and adds the new ones.
    """
    # 1. Clear existing documents without destroying the collection initialization
    existing_data = vectorstore.get()
    if existing_data["ids"]:
        vectorstore.delete(ids=existing_data["ids"])

    # 2. Load and split
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    # 3. Sanitize null bytes and filter empty strings
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
    chunks = [c for c in chunks if c.page_content.strip()]
    
    # 4. Batch processing (size=1 for stability with some providers)
    if chunks:
        for i in range(0, len(chunks), 1):
            vectorstore.add_documents(chunks[i:i + 1])
    
    return len(chunks)

def process_pdf():
    # Load environment variables from .env file
    load_dotenv()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define the path to the PDF file
    file_path = os.path.join(root_dir, "data", "Valve_NewEmployeeHandbook.pdf")
    
    if not os.access(root_dir, os.W_OK):
        db_path = os.path.join(tempfile.gettempdir(), "valve_db")
    else:
        db_path = os.path.join(root_dir, "valve_db")

    # Clean existing database directory to ensure a fresh build
    if os.path.exists(db_path):
        print(f"Cleaning existing database at {db_path}...")
        shutil.rmtree(db_path)

    try:
        print(f"Loading PDF from: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(pages)

        # Sanitize chunk content: Remove null bytes and filter empty strings
        # This prevents errors in the Google Embedding API mapping
        for chunk in chunks:
            chunk.page_content = chunk.page_content.replace("\x00", "")
        chunks = [c for c in chunks if c.page_content.strip()]
        
        print(f"Total valid chunks to process: {len(chunks)}")
        
        if not chunks:
            print("Error: No chunks were created. Verify the PDF content is readable.")
            return
            
        # Sanitize API Key
        raw_key = os.getenv("GOOGLE_API_KEY")
        try:
            if not raw_key and "GOOGLE_API_KEY" in st.secrets:
                raw_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass
            
        if not raw_key:
            print("Error: GOOGLE_API_KEY not found.")
            return
            
        api_key = raw_key.strip().strip('"').strip("'")

        # Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_document",
            google_api_key=api_key
        )

        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=db_path
        )
        
        # batch_size=1 is used to mitigate 'list index out of range' errors 
        # in the langchain-google-genai batch embedding implementation.
        batch_size = 1
        print(f"Storing documents in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            vectorstore.add_documents(chunks[i:i + batch_size])
        
        final_count = vectorstore._collection.count()
        print(f"Successfully stored {final_count} documents in '{db_path}'.")
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_pdf()
