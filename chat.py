import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def format_docs(docs):
    """Helper function to combine retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)

store = {}

def get_session_history(session_id: str):
    """Retrieve or create chat history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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
    
    # Define the prompt to reformulate the question based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Define the system prompt to guide the AI's behavior
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Build the retrieval and RAG chains
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever

    rag_chain = (
        RunnablePassthrough.assign(
            context=history_aware_retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap the chain with history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("Valve Handbook Chatbot with Memory is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        try:
            # Invoke the chain directly with the user string
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user_session_1"}}
            )
            print(f"\nAI: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
