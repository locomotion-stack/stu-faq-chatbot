import streamlit as st
import json

# Page config
st.set_page_config(page_title="STU FAQ Chatbot", page_icon="🎓", layout="centered")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load FAQ data
with open("faq.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logo and Title - Centered
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=80)
st.markdown("<h1 style='text-align: center; color: #0033A0;'>🎓 STU FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFB81C;'>Your AI Assistant for Sunyani Technical University</p>", unsafe_allow_html=True)

# Quick Question Buttons
st.write("**Quick Questions:**")
cols = st.columns(3)
questions = ["School Fees", "Course Registration", "Location"]
for i, q in enumerate(questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        # Find answer
        answer = "Sorry, I don't have an answer for that yet."
        for faq in faqs:
            if q.lower() in faq["question"].lower():
                answer = faq["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input - ALWAYS AT THE BOTTOM
if prompt := st.chat_input("Ask a question about STU..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Find answer
    answer = "Sorry, I couldn't find an answer. Try rephrasing or contact STU admin."
    for faq in faqs:
        if prompt.lower() in faq["question"].lower():
            answer = faq["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()