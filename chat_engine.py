import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# SET YOUR GEMINI API KEY
GEMINI_API_KEY = "Key"
genai.configure(api_key=GEMINI_API_KEY)

# Load FAISS index
def load_faiss_index(index_path="sebi_faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


# Load FinBERT model
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

# Perform sentiment classification
def classify_with_finbert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits).item()
    sentiments = ["positive", "negative", "neutral"]
    return sentiments[predicted_class_id]

# Enhance response using Gemini-Pro
import google.generativeai as genai

import google.generativeai as genai

def enhance_with_gemini(user_query, context, sentiment):
    # Configure the Gemini API using your API key
    genai.configure(api_key="KEY")  # Replace with your actual key

    # Correct model name: must match the tuned model ID from your console
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-001-tuning"
    )

    # Create the enhanced prompt
    prompt = f"""
    You are a highly intelligent Financial assistant. Only answer questions based on the content extracted from the provided PDF/document. Do not use any external or prior knowledge, and do not make assumptions. If the answer is not clearly present in the document, respond with: "The answer is not available in the provided document."

    Context:
    {context}

    Sentiment Analysis:
    {sentiment}

    User Query:
    {user_query}

    Generate an insightful, helpful response tailored to the query and sentiment.
    """

    # Generate response
    response = model.generate_content(prompt)

    return response.text


# Main chatbot logic
def answer_query(user_query, faiss_index, tokenizer, model):
    docs = faiss_index.similarity_search(user_query, k=3)
    combined_docs = "\n".join([doc.page_content for doc in docs])

    sentiment = classify_with_finbert(user_query, tokenizer, model)
    enhanced_response = enhance_with_gemini(user_query, combined_docs, sentiment)
    return enhanced_response
