import streamlit as st

st.set_page_config(
    page_title="HireSense AI | Interview Prep",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 HireSense AI: Professional Interview Preparation Platform")

# Clean Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Workspace")
    st.success("Platform status: **Operational**")
    st.divider()

st.markdown("""
### Welcome to your HireSense AI Workspace
This platform provides enterprise-grade resume screening, auto-ATS formatting, and realistic interview coaching powered by the Gemini 2.x LangChain pipeline.

👈 **Navigate the Platform:**
- **Dashboard & Optimizer**: Evaluate your resume against a Target JD, uncover skill gaps, and use the **Auto-Optimize** feature to bridge the gap!
- **AI Interview Coach**: Engage in a context-aware technical or behavioral interview based on your current resume.
""")
