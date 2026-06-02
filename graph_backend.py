import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # type: ignore
from langchain_core.documents import Document # type: ignore
from typing import List, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# 1. Define Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages in the current conversation.
        context: A string representing the retrieved context.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    context: str # Store formatted context string

def format_docs(docs):
    """Helper function to combine retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)

# Global initializations (or within main)
# It's better to initialize these once.
load_dotenv()

def get_sanitized_api_key():
    """Fetch API key from Streamlit secrets or environment and sanitize it."""
    # Streamlit secrets usually take precedence in production
    key = st.secrets.get("GOOGLE_API_KEY")
    
    # Fallback to .env for local development
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
        
    if not key:
        raise ValueError("GOOGLE_API_KEY is missing. Please set it in .env or Streamlit Secrets.")
    
    # Remove whitespace, quotes, and hidden carriage returns
    return str(key).strip().replace('"', '').replace("'", "").replace('\r', '')

api_key = get_sanitized_api_key()

# Define an absolute path for the vectorstore to prevent "readonly" issues
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ROOT_DIR, "valve_db")

# For the backend, we use retrieval_query because its main job is to 
# embed user questions for the retriever.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_query",
    google_api_key=api_key
)

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Prompts
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

# Sub-chain for history-aware retrieval
history_aware_retriever_standalone = contextualize_q_prompt | llm | StrOutputParser()

# 2. Define Nodes
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieves documents based on the latest user question,
    potentially reformulating it using chat history.
    """
    messages = state["messages"]
    last_message = messages[-1] # This should be the HumanMessage
    
    standalone_question = history_aware_retriever_standalone.invoke({
        "input": last_message.content,
        "chat_history": messages[:-1]
    })

    retrieved_documents = retriever.invoke(standalone_question)
    
    # Format the retrieved documents into a single string
    formatted_context = format_docs(retrieved_documents)
    
    # Update the state with the retrieved context
    return {"context": formatted_context}

def generate(state: GraphState) -> GraphState:
    """Generates an answer based on the retrieved context and chat history."""
    messages = state["messages"]
    context = state["context"]
    current_question = messages[-1].content
    
    answer_chain = qa_prompt | llm | StrOutputParser()
    
    response_content = answer_chain.invoke({
        "context": context,
        "chat_history": messages[:-1], # Previous messages
        "input": current_question # Current user question
    })
    
    # Append the AI's response as an AIMessage to the state's messages list.
    # LangGraph's state updates are additive for lists.
    return {"messages": [AIMessage(content=response_content)]}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)

def get_chatbot_response(user_input: str, thread_id: str = "default_session") -> str:
    """
    Invocates the LangGraph application with a user message 
    and returns the final AI response string.
    """
    try:
        final_state = rag_app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # Extract and return the last message content
        return final_state["messages"][-1].content
    except Exception as e:
        return f"An error occurred in the graph backend: {e}"
