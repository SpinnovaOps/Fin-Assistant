import streamlit as st
from db.db import init_db
from ui.onboarding import onboarding_form
from chat_engine import load_faiss_index, load_finbert, answer_query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator
import google.generativeai as genai

# Initialize DB
init_db()

# Configure Gemini-Pro API Key
genai.configure(api_key="AIzaSyCjeWAsyXA24ercu7XRISggxH0_Fzf68Kw")

# ---------------- Session Initialization ---------------- #

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'finbert_model' not in st.session_state:
    st.session_state.finbert_model = None

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat()

if 'user_info' not in st.session_state:
    st.session_state.user_info = {}

if 'language' not in st.session_state:
    st.session_state.language = None

if 'language_code' not in st.session_state:
    st.session_state.language_code = 'en'  # Default to English code

# Initialize translator
translator = Translator()

# ---------------- Load Models ---------------- #

if st.session_state.tokenizer is None or st.session_state.model is None:
    with st.spinner("Loading FinBERT model..."):
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        st.session_state.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

if st.session_state.faiss_index is None:
    st.session_state.faiss_index = load_faiss_index()

# ---------------- Regional Language Mapping ---------------- #

REGIONAL_LANGUAGES = {
    ("India", "Karnataka"): "Kannada",
    ("India", "Tamil Nadu"): "Tamil",
    ("India", "Maharashtra"): "Marathi",
    ("India", "West Bengal"): "Bengali",
    ("India", "Gujarat"): "Gujarati",
    ("India", "Kerala"): "Malayalam",
    ("India", "Telangana"): "Telugu",
    # Add more as needed
}

# Language code mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Telugu": "te",
    # Add more as needed
}

# ---------------- Onboarding ---------------- #

if not st.session_state.user_info:
    onboarding_form()
else:
    st.success("✅ You're onboarded!")

    country = st.session_state.user_info.get("country")
    state = st.session_state.user_info.get("state")

    # Language Selection
    language_options = ["English", "Hindi"]
    regional_lang = REGIONAL_LANGUAGES.get((country, state))
    if regional_lang:
        language_options.append(regional_lang)

    selected_language = st.selectbox(
        "🌐 Choose your preferred language:", 
        language_options,
        index=language_options.index(st.session_state.language) if st.session_state.language in language_options else 0
    )
    
    # Update language if changed
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.session_state.language_code = LANGUAGE_CODES.get(selected_language, 'en')

# ---------------- Chat Section ---------------- #

if st.session_state.language:
    st.header(f"💬 Chat in {st.session_state.language}")
    user_input = st.text_input("Ask your financial question:")

    if user_input:
        with st.spinner("Analyzing and Generating Response..."):
            # Step 1: Detect input language and translate to English if needed
            detected_lang = translator.detect(user_input).lang
            if detected_lang != 'en':
                input_english = translator.translate(user_input, src=detected_lang, dest='en').text
            else:
                input_english = user_input

            # Step 2: Process using FinBERT → FAISS → Gemini-Pro
            finbert_response = answer_query(
                input_english,
                st.session_state.faiss_index,
                st.session_state.tokenizer,
                st.session_state.model
            )

            # Step 3: Create prompt that ensures output matches user's language preference
            target_lang_code = st.session_state.language_code
            target_lang_name = st.session_state.language
            
            prompt = f"""
            Here is the financial context: {finbert_response}

            Please improve this response and return in bullet points.
            
            IMPORTANT: Your response must be in {target_lang_name} language only.
            """

            # Step 4: Generate response with Gemini-Pro that's already in the target language
            gemini_reply = st.session_state.gemini_chat.send_message(prompt).text
            
            # Step 5: Verify language and translate if necessary
            detected_output_lang = translator.detect(gemini_reply).lang
            
            # Only translate if the output isn't already in the requested language
            if detected_output_lang != target_lang_code:
                final_response = translator.translate(gemini_reply, src=detected_output_lang, dest=target_lang_code).text
            else:
                final_response = gemini_reply

            # Output
            st.markdown("### 🤖 Response:")
            st.markdown(final_response)
            
            # Display language confirmation
            st.caption(f"Response provided in {target_lang_name}")