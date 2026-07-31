import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE SETUP
st.set_page_config(page_title="STU AI Assistant", page_icon="🎓")
st.title("🎓 Sunyani Technical University AI Assistant")
st.caption("Ask me anything about admissions, fees, and programmes")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# 2. LOAD API KEY AND GEMINI CLIENT
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) # <-- just create client here

# 3. LOAD FAQ FROM JSON FILE
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

documents = [item["answer"] for item in faq_data["faqs"]]

# 4. SETUP TF-IDF SEARCH - NO TORCH!
@st.cache_resource
def load_vectorizer():
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = load_vectorizer()

# 5. AI BRAIN: RETRIEVE + GENERATE WITH GEMINI
def get_ai_answer(user_q):
    # RETRIEVE: Find top 3 most relevant FAQ chunks using TF-IDF
    q_vector = vectorizer.transform([user_q])
    similarities = cosine_similarity(q_vector, doc_vectors)
    top_indices = similarities.argsort()[0][-3:][::-1]
    context = "\n\n".join([documents[i] for i in top_indices])

    # GENERATE: Ask Gemini to answer
    prompt = f"""You are the friendly STU AI Assistant. Use ONLY the context below.
Be warm, conversational, and summarize. Don't just list.
If the answer is not in the context, say "I don't have that info, but contact admissions: 0352023278"

CONTEXT:
{context}

STUDENT QUESTION: {user_q}

ANSWER:"""

    # FIXED THIS LINE - use client.models.generate_content
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt
    )
    return response.text + "\n\n*Source: STU FAQ*"

# 6. STREAMLIT CHAT UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("**Quick Questions:**")
cols = st.columns(4)
questions = ["Master's Programmes", "Application Cost", "How to Apply", "STU Location"]
for i, q in enumerate(questions):
    if cols[i].button(q):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            answer = get_ai_answer(q)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if prompt := st.chat_input("Ask me anything about STU..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = get_ai_answer(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
