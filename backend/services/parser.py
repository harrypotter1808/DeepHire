import pdfplumber
import re
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Robustly extract text from a PDF file byte stream using pdfplumber.
    Handles formatting edge cases and missing sections.
    """
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return ""
    
    # Cleanup extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_resume_sections(text: str) -> dict:
    """
    Extract logic blocks from a raw resume text to facilitate component matching.
    """
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "full_text": text
    }
    
    # Heuristics based on common resume headers
    # Find Skills
    skills_match = re.search(r'(?i)(?:\bskills\b|\bcore competencies\b|technologies)(.*?)(?:\bexperience\b|\bemployment\b|\beducation\b|\bprojects\b|$)', text)
    if skills_match:
         sections["skills"] = skills_match.group(1).strip()

    # Find Experience
    exp_match = re.search(r'(?i)(?:\bexperience\b|\bemployment history\b|\bwork experience\b)(.*?)(?:\beducation\b|\bskills\b|\bprojects\b|$)', text)
    if exp_match:
         sections["experience"] = exp_match.group(1).strip()
         
    # Find Education
    edu_match = re.search(r'(?i)(?:\beducation\b|\bacademic background\b)(.*?)(?:\bskills\b|\bexperience\b|\bprojects\b|$)', text)
    if edu_match:
         sections["education"] = edu_match.group(1).strip()

    return sections
