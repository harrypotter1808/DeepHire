import re
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings

bert_model = None

def get_bert_model():
    global bert_model
    if bert_model is None:
        try:
            bert_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            print(f"Warning: Advanced Embeddings could not be loaded. {e}")
            bert_model = False
    return bert_model

def extract_keywords(text: str) -> set:
    """Extract alphabetic words > 4 chars as naive keywords for explainability"""
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    # Exclude common stop words manually to keep it lightweight
    stop_words = {'this', 'that', 'with', 'from', 'your', 'have', 'more', 'will', 'about'}
    return set(w for w in words if w not in stop_words)

def compute_tfidf_similarity(doc1: str, doc2: str) -> float:
    """Baseline Metric: TF-IDF + Cosine Similarity thresholding"""
    if not doc1 or not doc2:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([doc1, doc2])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def compute_bert_similarity(doc1: str, doc2: str) -> float:
    """Advanced Metric: Deep Embeddings similarity"""
    model = get_bert_model()
    if not doc1 or not doc2 or not model:
        return 0.0
    try:
        embeddings = model.embed_documents([doc1, doc2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def evaluate_resume(jd_text: str, resume_sections: dict) -> dict:
    """
    Computes multi-factor score, TF-IDF baseline, BERT advance, and Explainability variables.
    Score based on: Skill match (40%), Experience relevance (30%), Education (10%), Keywords (20%)
    """
    full_resume = resume_sections.get("full_text", "")
    
    # 1. Comparative Analysis (Baseline vs Advanced)
    tfidf_score = compute_tfidf_similarity(jd_text, full_resume)
    bert_score = compute_bert_similarity(jd_text, full_resume)
    
    # 2. Extract Explainability Data
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(full_resume)
    
    matched_keywords = list(jd_keywords.intersection(resume_keywords))
    missing_keywords = list(jd_keywords.difference(resume_keywords))
    keyword_score_ratio = len(matched_keywords) / max(len(jd_keywords), 1)

    # 3. Multi-factor Engine Scoring
    skills_sim = compute_bert_similarity(jd_text, resume_sections.get("skills", full_resume))
    exp_sim = compute_bert_similarity(jd_text, resume_sections.get("experience", full_resume))
    edu_sim = compute_bert_similarity(jd_text, resume_sections.get("education", full_resume))
    
    total_score = (
        (skills_sim * 40) +
        (exp_sim * 30) +
        (keyword_score_ratio * 20) +
        (edu_sim * 10)
    )

    return {
        "multi_factor_score": round(max(0, min(total_score, 100)), 2),
        "breakdown": {
            "skills": round(skills_sim * 40, 2),
            "experience": round(exp_sim * 30, 2),
            "keywords": round(keyword_score_ratio * 20, 2),
            "education": round(edu_sim * 10, 2),
        },
        "explainability": {
            "baseline_tfidf_similarity": round(tfidf_score, 3),
            "advanced_bert_similarity": round(bert_score, 3),
            "matched_keywords": matched_keywords[:8],
            "missing_keywords": missing_keywords[:8]
        }
    }
