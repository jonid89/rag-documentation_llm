from graph_backend import get_chatbot_response

# Test message 1
print("User: Hi, what is Valve?")
response1 = get_chatbot_response("Hi, what is Valve?", thread_id="test_1")
print(f"AI: {response1}\n")

# Test message 2 (Relies on memory/contextualization)
print("User: What is their handbook about?")
response2 = get_chatbot_response("What is their handbook about?", thread_id="test_1")
print(f"AI: {response2}")