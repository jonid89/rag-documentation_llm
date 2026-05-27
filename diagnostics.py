import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google import genai

def run_diagnostics(db_path="./valve_db"):
    """
    Performs a full system check:
    1. Environment variables
    2. LLM connectivity
    3. Embedding generation
    4. Model availability
    5. Local vectorstore integrity
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
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_query"
        )
        vector = embeddings.embed_query("Connectivity check.")
        print(f"SUCCESS: Generated embedding vector of length {len(vector)}")
    except Exception as e:
        print(f"FAILURE: Embedding connectivity check failed: {e}")

    print("\n--- 3. Listing All Models Available to your API Key ---")
    try:
        client = genai.Client(api_key=api_key)
        for m in client.models.list():
            actions = ", ".join(m.supported_actions) if m.supported_actions else "N/A"
            print(f"Model: {m.name} | Actions: [{actions}]")
    except Exception as e:
        print(f"FAILURE: Could not retrieve model list: {e}")

    print("\n--- 4. Checking Local Vectorstore State ---")
    if os.path.exists(db_path):
        print(f"SUCCESS: '{db_path}' directory found.")
        sqlite_path = os.path.join(db_path, "chroma.sqlite3")
        if os.path.exists(sqlite_path):
            print(f"SUCCESS: Database file 'chroma.sqlite3' is present.")
    else:
        print(f"WARNING: '{db_path}' directory NOT found. Run build_db.py to initialize.")

    print("\n--- Diagnostic check complete ---")

if __name__ == "__main__":
    run_diagnostics()