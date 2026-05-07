# streamlit_chatbot.py

import streamlit as st
import requests

# ---------------------
# Configuration
# ---------------------
# Replace this with the ngrok URL printed from your Colab FastAPI server
NGROK_URL = "https://perspiry-maris-obstetrically.ngrok-free.dev"  # <-- your ngrok URL
API_ENDPOINT = f"{NGROK_URL}/chat"

# ---------------------
# Streamlit page config
# ---------------------
st.set_page_config(page_title="Chatbot UI", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot")

# ---------------------
# Initialize chat history
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------
# Sidebar (optional settings)
# ---------------------
with st.sidebar:
    st.header("Settings")
    bot_name = st.text_input("Chatbot Name", value="Assistant")

# ---------------------
# User input
# ---------------------
user_input = st.text_input("You:", placeholder="Type your message here...")

# ---------------------
# Send message to API
# ---------------------
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Send request to FastAPI backend
        response = requests.post(API_ENDPOINT, json={"message": user_input})
        if response.status_code == 200:
            bot_response = response.json().get("response", "")
        else:
            bot_response = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # Append bot response
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# ---------------------
# Display chat history
# ---------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**{bot_name}:** {message['content']}")
