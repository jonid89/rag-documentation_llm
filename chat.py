import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Helper function to combine retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # Load environment variables (GOOGLE_API_KEY)
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return

    # Initialize the same embedding model used for database creation
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Load the existing Chroma vector store from the local directory
    vectorstore = Chroma(
        persist_directory="./valve_db",
        embedding_function=embeddings
    )

    # Initialize the Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Define the system prompt to guide the AI's behavior
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the retrieval chain using LCEL pipes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Valve Handbook Chatbot is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        try:
            # Invoke the chain directly with the user string
            response = rag_chain.invoke(user_input)
            print(f"\nAI: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
