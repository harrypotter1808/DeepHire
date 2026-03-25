import pytest
from backend.services.parser import parse_resume_sections

def test_parse_resume_sections_heuristics():
    """
    Evaluates the basic resume logical extraction heuristic.
    This acts as a proxy for evaluating parsing error handling on unstructured text.
    """
    sample_text = """
    JOHN DOE
    Skills Java, Python, Machine Learning, React, Streamlit
    Experience
    Software Engineer at TechCorp. Developed scalable pipelines.
    Education
    BS in Computer Science, State University
    """
    sections = parse_resume_sections(sample_text)
    
    assert "Java, Python" in sections["skills"], "Skills parsing precision failed"
    assert "Software Engineer" in sections["experience"], "Experience extraction precision failed"
    assert "Computer Science" in sections["education"], "Education structure missing"
