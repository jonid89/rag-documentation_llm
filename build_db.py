import os
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

    try:
        # Initialize the PyPDFLoader
        loader = PyPDFLoader(file_path)
        
        # Load the document
        # loader.load() returns a list of Document objects (one per page)
        pages = loader.load()
        
        # Initialize the RecursiveCharacterTextSplitter
        # This splitter is recommended for generic text as it tries to keep 
        # paragraphs, sentences, and words together as much as possible.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the documents into chunks
        chunks = text_splitter.split_documents(pages)
        print(f"Total chunks created: {len(chunks)}")

        # Clean and filter chunks: Remove null bytes and empty strings
        for chunk in chunks:
            chunk.page_content = chunk.page_content.replace("\x00", "")
        
        chunks = [c for c in chunks if c.page_content.strip()]
        
        if not chunks:
            print("Error: No chunks were created. Verify the PDF content is readable.")
            return

        # Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_document"
        )

        # Initialize Chroma and add documents in smaller batches
        # This prevents 'list index out of range' errors caused by API batch limits or safety filters
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./valve_db"
        )
        
        batch_size = 1
        for i in range(0, len(chunks), batch_size):
            current_batch = chunks[i:i + batch_size]
            print(f"Processing batch {(i // batch_size) + 1}...")
            vectorstore.add_documents(current_batch)
        
        # Verify count from the actual persistent collection
        print(f"Successfully stored {vectorstore._collection.count()} documents in 'valve_db'.")
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_pdf()
