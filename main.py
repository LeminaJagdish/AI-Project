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
    raise ValueError("Missing required environment variables.")

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

MODEL_NAME = "llama-3.3-70b-versatile"


# --------------------------------------------------
# Initialize Groq Client
# --------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------
# Initialize Astra DB
# --------------------------------------------------

astra_client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = astra_client.get_database(ASTRA_DB_API_ENDPOINT)
collection = db.get_collection("github_issues")

# --------------------------------------------------
# Local Embedding Model (No OpenAI)
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

# --------------------------------------------------
# Add Note to Astra DB (Run Once Then Comment)
# --------------------------------------------------

def add_note_to_db(note_text: str):
    embedding = get_embedding(note_text)

    collection.insert_one({
        "text": note_text,
        "$vector": embedding
    })

    print("✅ Note added successfully!")

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

    # If model calls a tool
    if hasattr(message, "tool_calls") and message.tool_calls:
 

        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "search_documents":
            tool_result = search_documents(arguments["query"])

        # Send tool result back to model
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
# Run Agent
# --------------------------------------------------

if __name__ == "__main__":

    # 🔥 Run once to insert notes, then comment out
    # add_note_to_db("Flash messages are temporary alerts in Flask applications.")
    # add_note_to_db("Vector databases store embeddings for semantic similarity search.")

    question = input("Ask a question: ")
    answer = agent(question)

    print("\n🤖 Agent Answer:\n")
    print(answer)
