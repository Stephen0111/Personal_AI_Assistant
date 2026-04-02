import streamlit as st
import requests, os, json
from datetime import datetime

st.set_page_config(
    page_title="AI Portfolio Assistant",
    page_icon="🤖",
    layout="wide"
)

AGENT_URL = os.environ.get("AGENT_FUNCTION_URL", "")

st.markdown("""

""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### I am visiting as a...")
    role = st.radio("", ["Recruiter","Engineer","Hiring Manager","Student"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Explainability mode")
    explain = st.toggle("Show retrieval reasoning", value=False)
    st.markdown("---")
    st.markdown("### Quick questions")
    for q in ["What is your tech stack?",
               "Tell me about your GCP projects",
               "How do I book a meeting?"]:
        if st.button(q, use_container_width=True):
            st.session_state.quick_q = q

st.title("Hi, I'm Stephen's AI assistant 👋")
st.caption("Ask me anything about his experience, projects, or to book a meeting.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything...")
if "quick_q" in st.session_state:
    prompt = st.session_state.pop("quick_q")

if prompt:
    st.session_state.messages.append(
        {"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(AGENT_URL, json={
                    "question": prompt,
                    "role": role.lower()
                }, timeout=60)
                data = resp.json()
                answer = data.get("answer","Sorry, I encountered an error.")
                intent = data.get("intent","general")

                st.markdown(answer)
                if explain:
                    st.caption(f"Intent classified as: **{intent}**")

                # Feedback
                col1, col2, _ = st.columns([1,1,8])
                with col1:
                    st.button("👍", key=f"up_{len(st.session_state.messages)}")
                with col2:
                    st.button("👎", key=f"dn_{len(st.session_state.messages)}")

            except Exception as e:
                answer = f"Connection error: {e}"
                st.error(answer)

    st.session_state.messages.append(
        {"role":"assistant","content":answer})
