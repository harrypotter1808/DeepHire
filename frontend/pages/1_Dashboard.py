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
                    result = resp.json()
                    
                    st.success("Analysis Complete!")
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
                         exp = result["explainability"]
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
                                    st.success("Resume Optimized Successfully! (-1 Credit)")
                                    with st.expander("View ATS-Optimized Resume", expanded=True):
                                        st.markdown(opt_resp.json()["optimized_resume"])
                                else:
                                    st.error("Failed to optimize resume.")
                            except Exception as opt_e:
                                st.error(f"Optimization API Connection failed: {opt_e}")
                    
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: Is the FastAPI backend running at {API_URL}? {e}")
