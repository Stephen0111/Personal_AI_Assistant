import streamlit as st
from dotenv import load_dotenv
from agents.controller import classify_intent
from agents.career import answer as career_answer
from agents.project_explainer import answer as project_answer
from analytics.tracker import log_event

load_dotenv()

st.set_page_config(page_title="AI Portfolio Assistant", page_icon="🤖", layout="wide")

# --- Sidebar: role selector ---
st.sidebar.title("Who are you?")
role = st.sidebar.selectbox(
    "Select your role for tailored responses:",
    ["general", "recruiter", "engineer", "hiring_manager"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**What I can help with:**")
st.sidebar.markdown("- My experience & skills\n- My projects & code\n- Booking a meeting")

# --- Main chat UI ---
st.title("👋 Hi, I'm [Your Name]'s AI Assistant")
st.caption("Ask me anything about my work, skills, or projects.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = classify_intent(prompt)
            log_event("query", {"intent": intent, "role": role})

            if intent == "career":
                response = career_answer(prompt, role)
            elif intent == "project":
                response = project_answer(prompt)
            elif intent == "scheduling":
                response = "To book a meeting, please use: [Calendly Link Here]. I'll confirm within 24 hours."
            else:
                response = "I'm best at answering questions about my experience and projects. Try asking about a specific project or skill!"

            st.markdown(response)

            # Feedback buttons
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                    log_event("feedback", {"sentiment": "positive"})
            with col2:
                if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                    log_event("feedback", {"sentiment": "negative"})

    st.session_state.messages.append({"role": "assistant", "content": response})
