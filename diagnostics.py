import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

def run_diagnostics():
    """
    Performs connectivity tests for LLM and Embeddings, and lists available models.
    Useful for troubleshooting 404, 403, or quota errors.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        return

    print("--- 1. Testing LLM Connectivity ('gemini-flash-latest') ---")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        response = llm.invoke("Hello, are you online?")
        print(f"SUCCESS: LLM responded: {response.content[:50]}...")
    except Exception as e:
        print(f"FAILURE: LLM connectivity check failed: {e}")

    print("\n--- 2. Testing Embedding Connectivity ('models/gemini-embedding-2') ---")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        vector = embeddings.embed_query("Diagnostic connectivity test.")
        print(f"SUCCESS: Generated embedding vector of length {len(vector)}")
    except Exception as e:
        print(f"FAILURE: Embedding connectivity check failed: {e}")

    print("\n--- 3. Listing All Models Available to your API Key ---")
    try:
        genai.configure(api_key=api_key)
        for m in genai.list_models():
            methods = ", ".join(m.supported_generation_methods)
            print(f"Model: {m.name} | Methods: [{methods}]")
    except Exception as e:
        print(f"FAILURE: Could not retrieve model list: {e}")

    print("\n--- Diagnostic check complete ---")

if __name__ == "__main__":
    run_diagnostics()