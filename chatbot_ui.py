import os
from pathlib import Path

import streamlit as st

from chatbot_api import ChatModel, ChatSession

# ====== INITIAL SETUP ======
st.set_page_config(page_title="Conversational Chatbot", page_icon="💬", layout="wide")
st.title("💬 Conversational Chatbot")


# ====== MODEL LOAD (cached) ======
@st.cache_resource
def load_session():
    chat = ChatSession(system_prompt="You are a helpful and concise assistant.")
    model = ChatModel(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    return chat, model


chat, model = load_session()

# ====== CHAT SELECTION ======
os.makedirs("chats", exist_ok=True)
chat_files = [f for f in os.listdir("chats") if f.endswith(".json")]

selected_file = st.selectbox(
    "Select a saved chat history:", options=["(New Chat)"] + chat_files, index=0
)

if selected_file != "(New Chat)" and (
    st.session_state.get("loaded_file") != selected_file
):
    chat.load_history(chat_id=Path(selected_file).stem)
    st.session_state.loaded_file = selected_file
    st.success(f"Loaded chat: {selected_file}")
elif selected_file == "(New Chat)" and st.session_state.get("loaded_file") is not None:
    chat.reset_history()
    st.session_state.loaded_file = None
else:
    pass

# ====== DISPLAYING CHAT HISTORY ======
st.session_state.chats = chat.get_history()
for turn in st.session_state.chats:
    message = turn["content"]
    role = turn["role"]
    if role != "system":
        with st.chat_message(role):
            st.markdown(message)

# ====== USER INPUT ======
if user_input := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    chat.add_message(role="user", content=user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        assistant_response = ""
        for t in model.generate(messages=st.session_state.chats, max_new_tokens=2048):
            assistant_response += t
            response_placeholder.markdown(assistant_response)

    chat.add_message(role="assistant", content=assistant_response)

    # Immediately saving chat history after each interaction
    chat.save_history()

    # Refreshing the page to update chat selection options
    st.rerun()
