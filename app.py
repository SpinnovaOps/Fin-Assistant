import streamlit as st
from db.db import init_db
from ui.onboarding import onboarding_form
from transformers import BertTokenizer, BertForSequenceClassification
from chat_engine import load_faiss_index
from vertexai.language_models import ChatModel
import streamlit as st

# Safely initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = None

if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

if 'language_selected' not in st.session_state:
    st.session_state.language_selected = False

# Initialize tokenizer
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Initialize FinBERT model
if 'finbert_model' not in st.session_state:
    st.session_state.finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Initialize FAISS index
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = load_faiss_index()
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT model and tokenizer if not already loaded
if st.session_state.tokenizer is None or st.session_state.model is None:
    with st.spinner("Loading FinBERT model..."):
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        st.session_state.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Initialize Gemini-Pro (ChatModel)
import google.generativeai as genai

# Initialize Gemini-Pro with API Key
genai.configure(api_key="Google API Key")

if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = genai.GenerativeModel('gemini-pro').start_chat()
  # Or your Gemini model ID

# Initialize DB
init_db()

# Run Onboarding
if "user_info" not in st.session_state:
    onboarding_form()
else:
    st.write("✅ You're onboarded. Chatbot will be here soon...")
from chat_engine import load_faiss_index, load_finbert, answer_query

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = load_faiss_index()
    st.session_state.tokenizer, st.session_state.model = load_finbert()

if "user_info" in st.session_state:
    st.header("💬 AI Financial Chatbot")
    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Analyzing..."):
            response = answer_query(
                user_input,
                st.session_state.faiss_index,
                st.session_state.tokenizer,
                st.session_state.model
            )
            st.markdown("### 🤖 Response:")
            st.markdown(response)
