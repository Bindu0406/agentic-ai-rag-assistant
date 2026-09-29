import streamlit as st
from src.graph import query_agent

st.set_page_config(
    page_title="Agentic AI eBook Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.caption("Powered by LangChain, Pinecone Vector DB, and Groq Llama/Qwen")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any question about the Agentic AI eBook."}
    ]

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new user input
if user_input := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating response..."):
            response = query_agent(user_input)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})