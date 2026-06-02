import os
import shutil
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

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

    # Define the path to the PDF file
    file_path = "data/Valve_NewEmployeeHandbook.pdf"
    db_path = "./valve_db"

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
