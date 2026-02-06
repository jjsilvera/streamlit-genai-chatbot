from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI 

# ==========================================================
# Configuration
# ==========================================================
load_dotenv()

st.set_page_config(page_title="Multi-model Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot")

# ==========================================================
# Provider and model options
# ==========================================================
model_options = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4.1"],
    "Gemini": ["gemini-2.5-flash"],
    "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "Ollama": ["qwen2.5:3b"]
}

# ==========================================================
# Provider and model
# ==========================================================
provider = st.selectbox("Select provider:", list(model_options.keys()))

# Detect whether the provider is available
provider_disponible = provider in ["Groq", "Gemini"]

# Model dropdown (disabled if not available)
model = st.selectbox(
    "Select model:",
    model_options[provider],
    disabled=not provider_disponible
)

# ==========================================================
# History initialization
# ==========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar historial del chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# FUNCTION TO CREATE THE MODEL ACCORDING TO THE SUPPLIER
# ==========================================================
def load_model(provider_name: str, model_name: str):
    """Returns the appropriate model according to the selected provider."""
    if provider_name == "Groq":
        return ChatGroq(model=model_name, temperature=0.0)
    elif provider_name == "Gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    elif provider_name in ["OpenAI", "Ollama"]:
        st.warning(f"🚫 Provider **{provider_name}** currently unavailable.")
        return None
    else:
        st.error("Unsupported provider.")
        return None

# Initialize the model selected
llm = load_model(provider, model)

# ==========================================================
# Input user and answer from the model
# ==========================================================
user_prompt = st.chat_input(
    "Write your question here..." if provider_disponible else "Provider unavailable",
    disabled=not provider_disponible
)

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    if llm:
        with st.spinner(f"Getting data from {provider} ({model})..."):
            response = llm.invoke(
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *st.session_state.chat_history
                ]
            )
            assistant_response = response.content

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# ==========================================================
# Clear chat
# ==========================================================
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


st.markdown("""
---
**Tips:**
- **Groq**, **Ollama**, and **Gemini** providers are currently active.
- The **OpenAI** provider will display a warning and disable inputs.
- **Ollama** must be used locally.
- Use the clear chat button to restart the conversation.

""")




