import uuid
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from services.parser import extract_text_from_pdf, parse_resume_sections
from services.matcher import evaluate_resume
from services.rag import AIInterviewCoach, generate_skill_gap_advice

load_dotenv()

# Setup API Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HireSense-API")

app = FastAPI(title="HireSense AI", description="Robust ML Pipeline for Resume Tracking & AI Interviews")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for AI Coaches (Production should use Redis)
session_coaches = {}

class MatchResponse(BaseModel):
    multi_factor_score: float
    breakdown: dict
    explainability: dict
    career_advice: str
    resume_text: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/match", response_model=MatchResponse)
async def match_resume_to_jd(
    jd_text: str = Form(...),
    resume: UploadFile = File(...)
):
    try:
        logger.info(f"Received match request for JD length {len(jd_text)}")
        pdf_bytes = await resume.read()
        raw_text = extract_text_from_pdf(pdf_bytes)
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract readable text from the provided PDF.")
            
        sections = parse_resume_sections(raw_text)
        evaluation = evaluate_resume(jd_text, sections)
        
        missing_keys = evaluation["explainability"]["missing_keywords"]
        advice = generate_skill_gap_advice(missing_keys) if missing_keys else "You meet all keyword requirements for this role!"
        
        logger.info(f"Match computed: Score {evaluation['multi_factor_score']}")
        
        return {
            "multi_factor_score": evaluation["multi_factor_score"],
            "breakdown": evaluation["breakdown"],
            "explainability": evaluation["explainability"],
            "career_advice": advice,
            "resume_text": raw_text
        }
    except Exception as e:
        logger.error(f"Error in /match endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview/start")
async def start_interview(
    jd_text: str = Form(...),
    resume_text: str = Form(...)
):
    try:
        logger.info("Initializing new AI Interview Coaching Session")
        session_id = str(uuid.uuid4())
        coach = AIInterviewCoach(resume_text, jd_text)
        
        if not coach.api_key:
            raise HTTPException(status_code=500, detail="Server config missing GEMINI_API_KEY. Set it in .env.")
            
        session_coaches[session_id] = coach
        
        # Prompt the AI to begin the interview
        initial_prompt = "Hello! Please introduce yourself quickly as the HireSense AI Interview Coach, and ask your first question based on my resume."
        reply = coach.interact(initial_prompt)
        
        return {"session_id": session_id, "reply": reply}
    except Exception as e:
        logger.error(f"FATAL ERROR starting interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Interview Engine Error: {str(e)}")

@app.post("/interview/chat")
async def chat_interview(request: ChatRequest):
    if request.session_id not in session_coaches:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
        
    logger.info(f"Processing chat for session {request.session_id}")
    coach = session_coaches[request.session_id]
    reply = coach.interact(request.message)
    return {"reply": reply}

class OptimizeRequest(BaseModel):
    resume_text: str
    jd_text: str
    missing_keywords: list

from services.rag import optimize_ats_resume

@app.post("/optimize")
async def optimize_resume_endpoint(request: OptimizeRequest):
    logger.info("Executing ONE-CLICK Auto-ATS Optimization")
    try:
        optimized_text = optimize_ats_resume(request.resume_text, request.jd_text, request.missing_keywords)
        return {"optimized_resume": optimized_text}
    except Exception as e:
        logger.error(f"Error optimizing resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
