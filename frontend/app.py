import streamlit as st

st.set_page_config(
    page_title="HireSense AI | SaaS Platform",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 HireSense AI: The Ultimate Interview Preparation SaaS")

# Mock SaaS Sidebar Placeholder
with st.sidebar:
    st.markdown("### 👤 My Account")
    st.info("**Plan:** Pro Tier 🌟")
    st.metric("Optimization Credits", "5")
    st.progress(0.5)
    st.button("🚀 Upgrade Plan", use_container_width=True)

st.markdown("""
### Welcome to your HireSense AI Workspace
This platform provides enterprise-grade resume screening, auto-ATS formatting, and realistic interview coaching powered by the Gemini 1.5 LangChain pipeline.

👈 **Navigate the SaaS Platform:**
- **Dashboard & Optimizer**: Evaluate your resume against a Target JD, uncover skill gaps, and use **1 Credit** to Auto-Optimize your bullet points!
- **AI Interview Coach**: Engage in a context-aware technical or behavioral interview based on your current resume.
""")
