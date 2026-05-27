import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

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

        # Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_document"
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
