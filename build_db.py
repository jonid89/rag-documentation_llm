import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

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

        # Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Create and persist the chunks in a local Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./valve_db"
        )

        print(f"Successfully created and stored {len(chunks)} chunks in 'valve_db'.")
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_pdf()
