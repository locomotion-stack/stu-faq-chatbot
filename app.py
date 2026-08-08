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
```python
# ============================================================
# ASK GEMINI WITH CONVERSATION MEMORY
# ============================================================

def get_ai_answer(user_question):

    # --------------------------------------------------------
    # 1. GET RECENT CONVERSATION HISTORY
    # --------------------------------------------------------
    #
    # Keep only the last 6 messages.
    # This prevents the prompt from becoming unnecessarily large.
    #

    recent_messages = st.session_state.get(
        "messages",
        []
    )[-6:]


    # --------------------------------------------------------
    # 2. CREATE CONVERSATION TEXT
    # --------------------------------------------------------

    conversation_history = ""

    for message in recent_messages:

        role = message["role"]

        if role == "user":
            conversation_history += (
                f"Student: {message['content']}\n"
            )

        elif role == "assistant":
            conversation_history += (
                f"Assistant: {message['content']}\n"
            )


    # --------------------------------------------------------
    # 3. CREATE A BETTER SEARCH QUERY
    # --------------------------------------------------------
    #
    # This is important for follow-up questions.
    #
    # Example:
    #
    # Previous:
    # "What master's programmes does STU offer?"
    #
    # Current:
    # "How much is it?"
    #
    # TF-IDF can struggle with "How much is it?"
    # because those words don't tell it what "it" means.
    #
    # We therefore include the recent conversation when
    # searching the FAQ database.
    #

    search_query = conversation_history + (
        f"\nStudent's latest question: {user_question}"
    )


    # --------------------------------------------------------
    # 4. RETRIEVE RELEVANT STU FAQ INFORMATION
    # --------------------------------------------------------

    context = retrieve_context(
        search_query,
        top_k=3
    )


    # --------------------------------------------------------
    # 5. CREATE GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the friendly AI Assistant for
Sunyani Technical University (STU) in Ghana.

You help students with questions about:

- Admissions
- Application procedures
- Programmes
- Fees
- Entry requirements
- Contact information
- STU services
- General university information

IMPORTANT RULES:

1. Use ONLY information contained in the STU FAQ CONTEXT.
2. You may use the CONVERSATION HISTORY to understand what
   the student means by follow-up questions.
3. Do NOT invent information.
4. Do NOT make up fees, programmes, dates, requirements,
   telephone numbers, email addresses, or locations.
5. If the answer is not contained in the FAQ context,
   clearly tell the student that you do not have that
   information.
6. If information is unavailable, tell the student to
   contact STU Admissions on 0352023278.
7. Be warm, friendly, and conversational.
8. Keep answers reasonably short and easy to understand.
9. Do not mention TF-IDF, retrieval, prompts, or the
   technical system.
10. Answer the student's LATEST question.
11. Use previous messages only when they help explain
    what the latest question refers to.

CONVERSATION HISTORY:

{conversation_history}

STU FAQ CONTEXT:

{context}

LATEST STUDENT QUESTION:

{user_question}

ANSWER:
"""


    # --------------------------------------------------------
    # 6. SEND REQUEST TO GEMINI
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )


        # ----------------------------------------------------
        # 7. CHECK RESPONSE
        # ----------------------------------------------------

        if response and response.text:

            return (
                response.text.strip()
                + "\n\n*Source: STU FAQ*"
            )

        return (
            "Sorry, I couldn't generate an answer right now."
        )


    # --------------------------------------------------------
    # 8. HANDLE GEMINI ERRORS
    # --------------------------------------------------------

    except errors.ClientError as e:

        if e.code == 429:

            return (
                "⚠️ **Gemini API quota exceeded.**\n\n"
                "Please check your Google API quota "
                "and billing settings."
            )

        elif e.code == 404:

            return (
                "⚠️ **Gemini model unavailable.**\n\n"
                f"The model `{MODEL_NAME}` is not available "
                "to this API project."
            )

        elif e.code in (401, 403):

            return (
                "⚠️ **Gemini API authentication error.**\n\n"
                "Please check your API key and permissions."
            )

        return f"⚠️ Gemini API error: {e}"


    # --------------------------------------------------------
    # 9. HANDLE OTHER ERRORS
    # --------------------------------------------------------

    except Exception as e:

        return f"⚠️ Something went wrong: {e}"
```




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
