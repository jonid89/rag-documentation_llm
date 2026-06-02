# Valve Handbook RAG Chatbot

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://rag-documentationllm-ojlywb8tzrbnag7cavasq5.streamlit.app/)

A Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit, LangGraph, and Google Gemini. This application allows users to ask questions about a provided document (defaulting to the Valve New Employee Handbook) or upload their own PDF documents to dynamically build a knowledge base and chat with their data.

## Features
- **Interactive Chat Interface**: Ask questions and get context-aware answers directly referencing the documents.
- **PDF Upload Support**: Upload your own PDF documents to instantly build a temporary vector database and query against your own data.
- **Multiple LLM Support**: Choose from various Google Gemini models (e.g., Gemini 3.1 Flash Lite, Gemini 3.5 Flash) via a simple dropdown menu.
- **LangGraph Backend**: Features a robust, graph-based workflow for document retrieval, context formulation, and text generation.
- **Chroma Vector Store**: Uses ChromaDB for fast and local document embedding storage.

## Live Demo
Check out the live application running on Streamlit Community Cloud: 
[**Launch Streamlit App**](https://rag-documentationllm-ojlywb8tzrbnag7cavasq5.streamlit.app/)

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jonid89/rag-documentation_llm.git
   cd rag-documentation_llm
   ```

2. **Install Dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```