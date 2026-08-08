import streamlit as st
from google import genai
from google.genai import errors
from dotenv import load_dotenv
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="STU AI Assistant",
    page_icon="🎓"
)

st.title("🎓 Sunyani Technical University AI Assistant")
st.caption("Ask me anything about admissions, fees, and programmes")


def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css("style.css")


# ============================================================
# 2. LOAD API KEY AND GEMINI CLIENT
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY was not found. "
        "Please check your .env file."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# 3. LOAD FAQ FROM JSON FILE
# ============================================================

try:
    with open("faq_data.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    documents = [
        item["answer"]
        for item in faq_data["faqs"]
    ]

except FileNotFoundError:
    st.error("faq_data.json was not found.")
    st.stop()

except (json.JSONDecodeError, KeyError):
    st.error("There is a problem with the format of faq_data.json.")
    st.stop()


# ============================================================
# 4. SETUP TF-IDF SEARCH
# ============================================================

@st.cache_resource
def load_vectorizer():
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)

    return vectorizer, doc_vectors


vectorizer, doc_vectors = load_vectorizer()


# ============================================================
# 5. AI BRAIN: RETRIEVE + GENERATE WITH GEMINI
# ============================================================
def get_ai_answer(user_q):

    # --------------------------------------------------------
    # RETRIEVE
    # Find the top 3 most relevant FAQ answers
    # --------------------------------------------------------

    q_vector = vectorizer.transform([user_q])

    similarities = cosine_similarity(
        q_vector,
        doc_vectors
    )

    top_indices = similarities.argsort()[0][-3:][::-1]

    context = "\n\n".join(
        [documents[i] for i in top_indices]
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    prompt = f"""
You are the friendly STU AI Assistant for
Sunyani Technical University.

Use ONLY the information provided in the CONTEXT below.

Be warm, conversational, helpful, and concise.
Summarize the information naturally instead of simply
listing it.

If the answer cannot be found in the context, say:

"I don't have that information, but you can contact
STU Admissions on 0352023278."

Do not invent information.

CONTEXT:
{context}

STUDENT QUESTION:
{user_q}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if response.text:
            return response.text + "\n\n*Source: STU FAQ*"

        return "Sorry, I couldn't generate an answer."

    except errors.ClientError as e:

        if e.code == 429:
            return (
                "⚠️ Gemini API quota exceeded. "
                "Please check your Google AI Studio/API "
                "quota and billing settings."
            )

        elif e.code == 404:
            return (
                "⚠️ The Gemini model is not available "
                "for this API project. Please check the "
                "available models for your API key."
            )

        return f"⚠️ Gemini API error: {e}"

    except Exception as e:
        return f"⚠️ Something went wrong: {e}"



# ============================================================
# 6. STREAMLIT CHAT UI
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 7. NORMAL CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask your question about STU..."
)

if user_question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            answer = get_ai_answer(user_question)

        st.markdown(answer)

    # Save answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# 8. QUICK QUESTIONS
# ============================================================

st.write("### Quick Questions")

cols = st.columns(4)

questions = [
    "Master's Programmes",
    "Application Cost",
    "How to Apply",
    "STU Location"
]


for i, question in enumerate(questions):

    if cols[i].button(question):

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate answer
        answer = get_ai_answer(question)

        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()
