import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Interview Coach", layout="centered", page_icon="🎤")
st.title("🎤 AI Interview Coach")

st.markdown("Practice your interview skills with our context-aware AI Coach powered by LangChain and RAG.")

if "resume_text" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("⚠️ Please evaluate a candidate in the Dashboard first to extract their Resume text and Job context.")
    st.stop()
    
# Initialize conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "session_id" not in st.session_state:
    with st.spinner("Initializing FAISS Vector Store and Generating First Question..."):
        try:
            resp = requests.post(f"{API_URL}/interview/start", data={
                "jd_text": st.session_state["jd_text"],
                "resume_text": st.session_state["resume_text"]
            })
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["session_id"] = data["session_id"]
                st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
            else:
                st.error(f"Error starting session: {resp.text}")
                st.stop()
        except requests.exceptions.ConnectionError:
            st.error(f"Backend connection failed. Is FastAPI running on {API_URL}?")
            st.stop()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Your Response..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            try:
                resp = requests.post(f"{API_URL}/interview/chat", json={
                    "session_id": st.session_state["session_id"],
                    "message": prompt
                })
                if resp.status_code == 200:
                     reply = resp.json()["reply"]
                     st.markdown(reply)
                     st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                     st.error(f"Failed to get response: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Lost connection to the backend server.")
