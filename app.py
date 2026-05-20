from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf():
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

        # Print the total number of chunks created
        print(f"Total number of chunks created: {len(chunks)}")
        
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_pdf()
