import os
import json
from dotenv import load_dotenv
from astrapy import DataAPIClient
from groq import Groq
import streamlit as st
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

if not GROQ_API_KEY or not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
    st.error("Missing required environment variables.")
    st.stop()

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

MODEL_NAME = "llama-3.3-70b-versatile"

# --------------------------------------------------
# Initialize Clients (Cached for Performance)
# --------------------------------------------------

@st.cache_resource
def init_clients():
    groq_client = Groq(api_key=GROQ_API_KEY)

    astra_client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    db = astra_client.get_database(ASTRA_DB_API_ENDPOINT)
    collection = db.get_collection("github_issues")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return groq_client, collection, embedding_model


groq_client, collection, embedding_model = init_clients()

# --------------------------------------------------
# Embedding Function
# --------------------------------------------------

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

# --------------------------------------------------
# Vector Search Tool
# --------------------------------------------------

def search_documents(query, k=3):
    query_embedding = get_embedding(query)

    results = collection.find(
        sort={"$vector": query_embedding},
        limit=k
    )

    return [doc["text"] for doc in results]

# --------------------------------------------------
# Tool Definition
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the knowledge base for relevant notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User question to search"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# --------------------------------------------------
# Agent Logic
# --------------------------------------------------

def agent(user_query):

    system_prompt = """
You are an AI assistant.

If the user's question requires information from stored notes,
use the search_documents tool.

If the question is general (like greetings),
respond normally without calling any tool.

If information is not found in notes, say you don't know.
"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if hasattr(message, "tool_calls") and message.tool_calls:

        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "search_documents":
            tool_result = search_documents(arguments["query"])

        second_response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "\n".join(tool_result)
                }
            ]
        )

        return second_response.choices[0].message.content

    return message.content

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Knowledge Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(("user", user_input))

    # Generate response
    with st.spinner("Thinking..."):
        response = agent(user_input)

    # Display assistant response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append(("assistant", response))
