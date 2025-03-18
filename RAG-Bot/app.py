import streamlit as st
import sqlite3
from bot import ChatAgent

# Initialize session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "Chat 1"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("🤖 Rice Try Bot")

# Create chatbot instance
chatbot = ChatAgent(session_id=st.session_state["session_id"])

# Display last 5 messages
# st.markdown("### 📝 Last 5 Messages")
chat_history = chatbot.get_last_5_chats()
# st.markdown(f"```\n{chat_history}\n```")

# Display chat history
# st.markdown("### 💬 Chat History")
for user_input, bot_response in st.session_state["chat_history"]:
    st.markdown(f"🧑‍💻 **User:** {user_input}")
    st.markdown(f"🤖 **Bot:** {bot_response}")
    st.write("---")

# Get user input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = chatbot.chat(prompt)
        st.markdown(response)
    
    # Store chat
    st.session_state["chat_history"].append((prompt, response))
