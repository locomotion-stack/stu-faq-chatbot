import streamlit as st
from google import genai
from google.genai import errors
from dotenv import load_dotenv

import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="STU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Sunyani Technical University AI Assistant")

st.caption(
    "Ask me anything about admissions, fees, programmes, "
    "application procedures, and STU."
)


# ============================================================
# 2. LOAD CSS
# ============================================================

def load_css(file_name):

    try:

        with open(
            file_name,
            "r",
            encoding="utf-8"
        ) as file:

            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    except FileNotFoundError:

        # CSS is optional.
        # The application continues without it.
        pass


load_css("style.css")


# ============================================================
# 3. LOAD GOOGLE API KEY
# ============================================================

def get_api_key():

    # --------------------------------------------------------
    # STREAMLIT CLOUD
    # --------------------------------------------------------

    try:

        api_key = st.secrets["GOOGLE_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass


    # --------------------------------------------------------
    # LOCAL COMPUTER
    # --------------------------------------------------------

    load_dotenv()

    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    return api_key


API_KEY = get_api_key()


if not API_KEY:

    st.error(
        "❌ GOOGLE_API_KEY is missing.\n\n"
        "For Streamlit Cloud, add it under:\n\n"
        "Manage app → Settings → Secrets\n\n"
        "Use:\n\n"
        'GOOGLE_API_KEY = "your_api_key_here"'
    )

    st.stop()


# ============================================================
# 4. CREATE GEMINI CLIENT
# ============================================================

try:

    client = genai.Client(
        api_key=API_KEY
    )

except Exception as e:

    st.error(
        f"❌ Could not create Gemini client:\n\n{e}"
    )

    st.stop()


# ============================================================
# 5. GEMINI MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# 6. LOAD FAQ DATA
# ============================================================

try:

    with open(
        "faq_data.json",
        "r",
        encoding="utf-8"
    ) as file:

        faq_data = json.load(file)

except FileNotFoundError:

    st.error(
        "❌ faq_data.json was not found.\n\n"
        "Make sure faq_data.json is in the same "
        "GitHub repository/folder as app.py."
    )

    st.stop()

except json.JSONDecodeError:

    st.error(
        "❌ faq_data.json contains invalid JSON."
    )

    st.stop()


# ============================================================
# 7. EXTRACT FAQS
# ============================================================

try:

    faqs = faq_data["faqs"]

except (KeyError, TypeError):

    st.error(
        "❌ Your faq_data.json must contain a "
        "'faqs' list."
    )

    st.stop()


if not faqs:

    st.error(
        "❌ No FAQs were found in faq_data.json."
    )

    st.stop()


# ============================================================
# 8. PREPARE FAQ DOCUMENTS
# ============================================================

documents = []

for item in faqs:

    if isinstance(item, dict):

        answer = item.get(
            "answer",
            ""
        )

        if answer:

            documents.append(
                answer
            )


if not documents:

    st.error(
        "❌ No FAQ answers were found."
    )

    st.stop()


# ============================================================
# 9. CREATE TF-IDF SEARCH ENGINE
# ============================================================

@st.cache_resource
def create_search_engine(documents):

    vectorizer = TfidfVectorizer(
        lowercase=True
    )

    document_vectors = vectorizer.fit_transform(
        documents
    )

    return (
        vectorizer,
        document_vectors
    )


vectorizer, doc_vectors = create_search_engine(
    documents
)


# ============================================================
# 10. RETRIEVE RELEVANT FAQ INFORMATION
# ============================================================

def retrieve_context(
    search_query,
    top_k=3
):

    # --------------------------------------------------------
    # Convert question to TF-IDF vector
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [search_query]
    )


    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        doc_vectors
    )[0]


    # --------------------------------------------------------
    # Make sure top_k is valid
    # --------------------------------------------------------

    top_k = min(
        top_k,
        len(documents)
    )


    # --------------------------------------------------------
    # Find highest scoring FAQs
    # --------------------------------------------------------

    top_indices = (
        similarities
        .argsort()[-top_k:][::-1]
    )


    selected_documents = []


    # --------------------------------------------------------
    # Select relevant documents
    # --------------------------------------------------------

    for index in top_indices:

        score = similarities[index]

        if score > 0:

            selected_documents.append(
                documents[index]
            )


    # --------------------------------------------------------
    # If no FAQ matched
    # --------------------------------------------------------

    if not selected_documents:

        selected_documents = documents[:top_k]


    # --------------------------------------------------------
    # Combine FAQ information
    # --------------------------------------------------------

    return "\n\n".join(
        selected_documents
    )


# ============================================================
# 11. BUILD CONVERSATION HISTORY
# ============================================================

def get_conversation_history():

    messages = st.session_state.get(
        "messages",
        []
    )


    # Keep the last 6 messages.
    #
    # Example:
    #
    # User
    # Assistant
    # User
    # Assistant
    # User
    # Assistant

    recent_messages = messages[-6:]


    history = []


    for message in recent_messages:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )


        if role == "user":

            history.append(
                f"Student: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )


    return "\n".join(history)


# ============================================================
# 12. GENERATE AI ANSWER
# ============================================================

def get_ai_answer(user_question):

    # --------------------------------------------------------
    # GET PREVIOUS CONVERSATION
    # --------------------------------------------------------

    conversation_history = (
        get_conversation_history()
    )


    # --------------------------------------------------------
    # CREATE SEARCH QUERY
    #
    # This is what allows follow-up questions to work.
    #
    # Example:
    #
    # Student:
    # "What master's programmes does STU offer?"
    #
    # Student:
    # "How much is it?"
    #
    # The second search also sees the first question.
    # --------------------------------------------------------

    search_query = (
        conversation_history
        + "\n"
        + "Latest student question: "
        + user_question
    )


    # --------------------------------------------------------
    # RETRIEVE STU INFORMATION
    # --------------------------------------------------------

    context = retrieve_context(
        search_query,
        top_k=3
    )


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the friendly AI Assistant for
Sunyani Technical University (STU) in Ghana.

Your purpose is to help students with questions about:

- Admissions
- Application procedures
- Programmes
- Fees
- Entry requirements
- Application costs
- Contact information
- University location
- STU services
- General STU information

IMPORTANT INSTRUCTIONS:

1. Use ONLY the information contained in the
   STU FAQ CONTEXT.

2. Use the CONVERSATION HISTORY only to understand
   follow-up questions.

3. Always answer the student's LATEST question.

4. If the student says something like:
   "How much is it?"
   "What are the requirements?"
   "When does it start?"
   use the previous conversation to understand
   what they are referring to.

5. NEVER invent information.

6. NEVER make up:
   - Fees
   - Programmes
   - Admission dates
   - Requirements
   - Phone numbers
   - Email addresses
   - Locations
   - Application procedures

7. If the answer is not available in the FAQ context,
   respond:

   "I don't have that information, but you can contact
   STU Admissions on 0352023278."

8. Be warm, friendly, and conversational.

9. Keep answers concise and easy for students
   to understand.

10. Do not mention:
    - TF-IDF
    - Retrieval
    - Prompt
    - Context
    - API
    - Technical implementation

11. Do not say you searched a database.

CONVERSATION HISTORY:

{conversation_history}

STU FAQ INFORMATION:

{context}

LATEST STUDENT QUESTION:

{user_question}

ANSWER:
"""


    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )


        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if response and response.text:

            return (
                response.text.strip()
                + "\n\n*Source: STU FAQ*"
            )


        return (
            "Sorry, I couldn't generate an answer "
            "right now."
        )


    # ========================================================
    # GEMINI API ERRORS
    # ========================================================

    except errors.ClientError as e:

        # ----------------------------------------------------
        # 429 = QUOTA
        # ----------------------------------------------------

        if e.code == 429:

            return (
                "⚠️ **Gemini API quota exceeded.**\n\n"
                "Please check your Google API quota "
                "and billing settings."
            )


        # ----------------------------------------------------
        # 404 = MODEL NOT AVAILABLE
        # ----------------------------------------------------

        elif e.code == 404:

            return (
                "⚠️ **Gemini model unavailable.**\n\n"
                f"The model `{MODEL_NAME}` is not "
                "available to this API project."
            )


        # ----------------------------------------------------
        # 401 / 403 = API KEY PROBLEM
        # ----------------------------------------------------

        elif e.code in (401, 403):

            return (
                "⚠️ **Gemini API authentication error.**\n\n"
                "Please check your GOOGLE_API_KEY."
            )


        # ----------------------------------------------------
        # OTHER GEMINI ERROR
        # ----------------------------------------------------

        return (
            f"⚠️ Gemini API error: {e}"
        )


    # ========================================================
    # OTHER PYTHON ERRORS
    # ========================================================

    except Exception as e:

        return (
            f"⚠️ Something went wrong: {e}"
        )


# ============================================================
# 13. INITIALIZE CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 14. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    with st.chat_message(role):

        st.markdown(
            content
        )


# ============================================================
# 15. NORMAL CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about STU..."
)


if user_question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # Save user message BEFORE calling AI
    #
    # This is important because get_ai_answer()
    # reads the conversation history.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            answer = get_ai_answer(
                user_question
            )

        st.markdown(
            answer
        )


    # --------------------------------------------------------
    # Save assistant answer
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# 16. QUICK QUESTIONS
# ============================================================

st.write("### 💡 Quick Questions")


col1, col2 = st.columns(2)

col3, col4 = st.columns(2)


quick_questions = [
    (col1, "🎓 Master's Programmes"),
    (col2, "💰 Application Cost"),
    (col3, "📝 How to Apply"),
    (col4, "📍 STU Location")
]


for column, display_question in quick_questions:

    if column.button(
        display_question,
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Remove emoji from question
        # ----------------------------------------------------

        question = display_question.split(
            " ",
            1
        )[1]


        # ----------------------------------------------------
        # Save student question
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = get_ai_answer(
            question
        )


        # ----------------------------------------------------
        # Save assistant answer
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # ----------------------------------------------------
        # Refresh Streamlit
        # ----------------------------------------------------

        st.rerun()


# ============================================================
# 17. CLEAR CHAT BUTTON
# ============================================================

st.divider()


if st.button(
    "🗑️ Clear Conversation",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


# ============================================================
# 18. FOOTER
# ============================================================

st.caption(
    "🎓 STU AI Assistant • "
    "Powered by the STU FAQ knowledge base"
)
