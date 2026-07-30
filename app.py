import streamlit as st
import json

st.set_page_config(page_title="STU Chatbot", page_icon="🎓", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# ===== 1. LOAD FAQs =====
@st.cache_data
def load_faqs():
    with open('faq.json', 'r', encoding='utf-8') as f:
        return json.load(f)

faqs = load_faqs()

# ===== 2. SMART SEARCH FUNCTION =====
def find_answer(user_q):
    user_q = user_q.lower()

    for faq in faqs:
        q = faq["question"].lower()
        a = faq["answer"].lower()

        # Check if any key words from user match the FAQ question
        user_words = [word for word in user_q.split() if len(word) > 3]
        if any(word in q for word in user_words):
            return faq["answer"]

        # Keyword shortcuts
        if any(word in user_q for word in ["location", "where", "address"]):
            if "sunyani" in a or "bono" in a:
                return faq["answer"]
        if any(word in user_q for word in ["fee", "fees", "cost", "price", "ghc"]):
            if "gh¢" in a:
                return faq["answer"]
        if any(word in user_q for word in ["admission", "apply", "voucher", "form"]):
            if "admissions.stu.edu.gh" in a or "voucher" in a:
                return faq["answer"]
        if any(word in user_q for word in ["programme", "program", "course", "degree"]):
            if "btech" in a or "mtech" in a or "hnd" in a:
                return faq["answer"]

    return "Sorry, I couldn't find an answer for that. 😅\n\nTry asking about: **Programmes, Admissions, Fees**\n\nOr contact STU Admissions: 0352023278, 0501512556"

# ===== 3. CHAT UI =====
st.title("🎓 Sunyani Technical University Chatbot")
st.caption("Ask me anything about STU admissions, programmes, and fees")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm the STU Assistant. How can I help you today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Question Buttons
st.write("**Quick Questions:**")
cols = st.columns(4)
quick_questions = ["Master's Programmes", "Application Cost", "How to Apply", "STU Location"]

for i, q in enumerate(quick_questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        answer = find_answer(q)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# User input
if prompt := st.chat_input("Ask me anything about STU..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = find_answer(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
