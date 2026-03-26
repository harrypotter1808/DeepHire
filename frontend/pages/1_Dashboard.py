import streamlit as st
import requests

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Comparison Dashboard", layout="wide")
st.title("📊 A/B Comparison Dashboard")

st.markdown("Upload a resume and provide a Job Description to evaluate the candidate's match using our Multi-Factor Scoring Engine.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area("Paste Job Description here", height=250)

with col2:
    st.subheader("Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Evaluate Candidate", type="primary"):
    if not jd_input or not uploaded_file:
        st.error("Please provide both JD and Resume.")
    else:
        with st.spinner("Analyzing with TF-IDF Baseline and BERT Models..."):
            try:
                files = {"resume": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"jd_text": jd_input}
                resp = requests.post(f"{API_URL}/match", files=files, data=data)
                
                if resp.status_code == 200:
                    st.session_state["evaluation_result"] = resp.json()
                    st.session_state["jd_input"] = jd_input
                    st.success("Analysis Complete!")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# Render results from session state (Persistent across re-runs)
if "evaluation_result" in st.session_state:
    result = st.session_state["evaluation_result"]
    jd_input = st.session_state["jd_input"]
    exp = result["explainability"]
    
    # Store context in session state for Interview Bot
    st.session_state["resume_text"] = result["resume_text"]
    st.session_state["jd_text"] = jd_input
    
    # Display Overall Score
    st.subheader("🎯 Overall Match Score")
    st.progress(result["multi_factor_score"] / 100.0)
    st.markdown(f"**{result['multi_factor_score']}%** multi-factor fit.")
    
    st.divider()
    
    # Comparison Metrics
    c1, c2 = st.columns(2)
    with c1:
         st.markdown("### Model Comparison")
         st.metric("Baseline (TF-IDF Similarity)", f"{int(exp['baseline_tfidf_similarity'] * 100)}%")
         st.metric("Advanced (BERT Dense Embeddings)", f"{int(exp['advanced_bert_similarity'] * 100)}%")
         st.caption("BERT handles semantic mismatches much better than exact keyword frequencies.")
         
    with c2:
         st.markdown("### Scoring Engine Breakdown (Weights)")
         st.json(result["breakdown"])
         
    st.divider()
    
    # Explainability
    st.markdown("### 🧠 Explainability Analysis")
    k1, k2 = st.columns(2)
    k1.success(f"Matched Keywords: {', '.join(exp['matched_keywords'])}")
    k2.error(f"Missing Keywords: {', '.join(exp['missing_keywords'])}")
    
    st.divider()
    st.markdown("### 🚀 AI Career Advisor & Learning Recommendation Engine")
    with st.container():
        st.markdown(result.get("career_advice", "No explicit career advice generated."))
        
    st.divider()
    st.markdown("### ✨ One-Click ATS Optimization (SaaS Feature)")
    st.info("Uses **1 Pro Credit** to intelligently rewrite your experience bullets, incorporating missing keywords and strict STAR methodology.")
    
    if st.button("🚀 Auto-Optimize My Resume", type="primary"):
        with st.spinner("Rewriting your resume with Gemini AI..."):
            try:
                opt_resp = requests.post(f"{API_URL}/optimize", json={
                    "resume_text": result["resume_text"],
                    "jd_text": jd_input,
                    "missing_keywords": exp['missing_keywords']
                })
                if opt_resp.status_code == 200:
                    st.session_state["optimized_resume"] = opt_resp.json()["optimized_resume"]
                    st.success("Resume Optimized Successfully! (-1 Credit)")
                else:
                    st.error("Failed to optimize resume.")
            except Exception as opt_e:
                st.error(f"Optimization API Connection failed: {opt_e}")

    if "optimized_resume" in st.session_state:
        with st.expander("View ATS-Optimized Resume", expanded=True):
            st.markdown(st.session_state["optimized_resume"])
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download Markdown (.md)", 
                                 st.session_state["optimized_resume"], 
                                 file_name="optimized_resume.md")
            
            with c2:
                if st.button("🎨 Convert to Professional LaTeX"):
                    with st.spinner("Translating to LaTeX..."):
                        try:
                            lat_resp = requests.post(f"{API_URL}/convert/latex", json={
                                "resume_text": st.session_state["optimized_resume"]
                            })
                            if lat_resp.status_code == 200:
                                st.session_state["latex_source"] = lat_resp.json()["latex_source"]
                                st.success("LaTeX Generated!")
                            else:
                                st.error("LaTeX Conversion failed.")
                        except Exception as lat_e:
                            st.error(f"LaTeX API Connection failed: {lat_e}")

    if "latex_source" in st.session_state:
        with st.expander("View LaTeX Source", expanded=True):
            st.code(st.session_state["latex_source"], language="latex")
            st.download_button("📥 Download LaTeX (.tex)", 
                             st.session_state["latex_source"], 
                             file_name="resume.tex")
