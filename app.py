import streamlit as st
import google.generativeai as genai
import pdfplumber  # Faster and more reliable than PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Gemini Setup ---
@st.cache_resource
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("models/gemini-1.5-flash")
    except Exception as e:
        st.error(f"🔴 Gemini Error: {str(e)}")
        return None

# --- PDF Processing ---
@st.cache_data
def load_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"  # Handle None returns
    return text

def chunk_text(text, max_words=150):  # Word-based chunking
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_words:  # +1 for space
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- TF-IDF Retrieval ---
@st.cache_data
def tfidf_indexing(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_similar_chunks(query, vectorizer, tfidf_matrix, chunks, top_k=3):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(cosine_sim)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Streamlit UI ---
st.set_page_config(page_title="🕉️ Spiritual Guru", layout="centered")
st.title("📚 Bhagavad Gita & Ramayana Wisdom")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar controls
with st.sidebar:
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    if api_key and not st.session_state.model:
        with st.spinner("Connecting to Gemini..."):
            st.session_state.model = configure_gemini(api_key)
            if st.session_state.model:
                st.success("API Connected!")
    
    uploaded_file = st.file_uploader("📜 Upload Scripture (PDF)", type="pdf")
    agent_type = st.selectbox("🧙 Guru Type", 
                           ["Gita Expert", "Ramayana Scholar", "Vedanta Teacher"])

# Main app logic
if uploaded_file and st.session_state.model:
    with st.spinner("🧘 Processing scripture..."):
        text = load_pdf_text(uploaded_file)
        chunks = chunk_text(text)
        vectorizer, tfidf = tfidf_indexing(chunks)
    
    question = st.text_area("🙏 Ask your spiritual question:", height=100)
    
    if question and st.button("Get Wisdom"):
        with st.spinner("🔍 Finding answers..."):
            try:
                relevant_chunks = find_similar_chunks(question, vectorizer, tfidf, chunks)
                prompt = f"""You are a {agent_type}. Answer concisely (3-4 lines max) in simple English.
Start with "Namaste seeker..." and end with a unique reflective question.

Question: {question}

Context:\n""" + "\n".join(f"- {chunk}" for chunk in relevant_chunks)
                
                response = st.session_state.model.generate_content(prompt)
                st.markdown(f"## 🪔 {agent_type.split()[0]}'s Answer")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👉 Please upload a PDF and enter API key")
