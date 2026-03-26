import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def select_best_chat_model(api_key):
    """Dynamically discover the best available Chat/Generation model for this API key."""
    try:
        genai.configure(api_key=api_key)
        gen_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority mapping: Try to find the highest-tier Flash/Pro models available
        priorities = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-flash-latest', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for p in priorities:
            match = next((m for m in gen_models if p in m), None)
            if match:
                return match
        return gen_models[0] if gen_models else "models/gemini-1.5-flash"
    except Exception:
        return "models/gemini-1.5-flash"

class AIInterviewCoach:
    """
    RAG-based Interview Coach utilizing LangChain, FAISS, and Gemini 1.5 Flash.
    Provides context-aware questions and constructive feedback based on resumes.
    """
    
    def __init__(self, resume_text: str, jd_text: str):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.chain = None
        self.chat_history = []
        
        if self.api_key:
            self._setup_rag_pipeline(resume_text, jd_text)
            
    def _setup_rag_pipeline(self, resume_text: str, jd_text: str):
        # 1. Document Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs = splitter.create_documents([
            f"--- JOB DESCRIPTION ---\n{jd_text}\n\n--- CANDIDATE RESUME ---\n{resume_text}"
        ])
        
        # 2. Vector Store Setup (FAISS) - DYNAMIC DISCOVERY
        genai.configure(api_key=self.api_key)
        all_models = list(genai.list_models())
        embedding_models = [m.name for m in all_models if any('embed' in method.lower() for method in m.supported_generation_methods)]
        
        if not embedding_models:
            raise ValueError("No compatible embedding models found for this Gemini API key.")
        
        # Prioritized list: Try text-embedding-004, then gemini-embedding-001, then embedding-001
        prioritized = ['text-embedding-004', 'gemini-embedding-001', 'embedding-001']
        best_model = None
        for p in prioritized:
            best_model = next((m for m in embedding_models if p in m), None)
            if best_model: break
            
        # Still no match in priorities? Use the first available embedding model
        best_model = best_model or embedding_models[0]
        
        embeddings = GoogleGenerativeAIEmbeddings(model=best_model, google_api_key=self.api_key)
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # 3. LLM Configuration
        best_chat_model = select_best_chat_model(self.api_key)
        llm = ChatGoogleGenerativeAI(model=best_chat_model, temperature=0.7, google_api_key=self.api_key)
        
        # 4. Prompt Engineering for AI Coach Profile
        system_prompt = (
            "You are an expert AI Technical Interview Coach for the product 'HireSense AI'.\n"
            "Your objective is to interview the candidate based on the Job Description and their Resume.\n"
            "Rules:\n"
            "1. Ask EXACTLY ONE question at a time.\n"
            "2. If the user is answering a previous question, evaluate it first. Give direct, constructive feedback (e.g., 'You lacked clarity on system design...', or 'Great answer, but you missed X...').\n"
            "3. Ask the next logical follow-up question, adjusting difficulty based on their response.\n\n"
            "Context from JD and Resume:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 5. Chain Construction
        qa_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(retriever, qa_chain)

    def interact(self, user_message: str) -> str:
        """Process a message and return the Coach's response"""
        if not self.chain:
             return "Please set GEMINI_API_KEY in the .env file to enable the AI Interview Coach."
             
        # The new create_retrieval_chain passes context implicitly from retriever
        # We could pass chat_history if needed, but for simplicity, the prompt handles the immediate input.
        try:
            response = self.chain.invoke({
                "input": user_message
            })
            answer = response["answer"]
            self.chat_history.append((user_message, answer))
            return answer
        except Exception as e:
            return f"An error occurred communicating with the AI Model: {str(e)}"

def generate_skill_gap_advice(missing_keywords: list) -> str:
    """Uses Gemini LLM to generate learning resources for missing keywords."""
    if not os.getenv("GEMINI_API_KEY") or not missing_keywords:
        return "Set GEMINI_API_KEY in the backend to unlock AI-powered learning suggestions."
        
    try:
        from langchain_core.prompts import PromptTemplate
        api_key = os.getenv("GEMINI_API_KEY")
        best_model = select_best_chat_model(api_key)
        llm = ChatGoogleGenerativeAI(model=best_model, temperature=0.5, google_api_key=api_key)
        prompt = PromptTemplate.from_template(
            "You are an AI Career Advisor forming a 'Learning Recommendation Engine'.\n"
            "The candidate is actively missing these key skills: {keywords}.\n"
            "Start by clearly stating: 'You are missing: [skills]'.\n"
            "Then, output a highly personalized learning path in Markdown format with these exact three headers:\n"
            "### 📚 Recommended Courses\n"
            "(Suggest 2 specific, high-quality courses from platforms like Coursera/Udemy/Pluralsight)\n\n"
            "### 🛠️ Projects to Build\n"
            "(Suggest 2 concrete, portfolio-worthy project ideas that practice these exact missing skills)\n\n"
            "### 🏆 Certifications\n"
            "(Suggest 1-2 relevant industry certifications that validate these missing skills)"
        )
        chain = prompt | llm
        response = chain.invoke({"keywords": ", ".join(missing_keywords)})
        return response.content
    except Exception as e:
        return f"Could not generate career advice: {e}"

def optimize_ats_resume(resume_text: str, jd_text: str, missing_keywords: list) -> str:
    """Uses Gemini LLM to completely rewrite and ATS-optimize the candidate's resume."""
    if not os.getenv("GEMINI_API_KEY"):
        return "Set GEMINI_API_KEY to unlock the One-Click Auto-ATS Optimizer."
        
    try:
        from langchain_core.prompts import PromptTemplate
        api_key = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6, google_api_key=api_key)
        prompt = PromptTemplate.from_template(
            "You are an elite Tech Recruiter and ATS Optimization Expert.\n"
            "Given the candidate's base resume below, REWRITE their experience and skills to strictly align with the provided Target Job Description.\n\n"
            "CRITICAL RULES:\n"
            "1. Naturally weave the following missing keywords into their experience bullet points (only if logically possible without lying): {keywords}.\n"
            "2. Ensure extremely impactful 'STAR' method formatting for all bullets (e.g. 'Achieved X by implementing Y resulting in Z').\n"
            "3. Return ONLY the rewritten, production-ready Resume in clean Markdown format.\n\n"
            "--- TARGET JOB DESCRIPTION ---\n{jd}\n\n"
            "--- CANDIDATE RESUME ---\n{resume}"
        )
        chain = prompt | llm
        response = chain.invoke({
            "keywords": ", ".join(missing_keywords) if missing_keywords else "None missing",
            "jd": jd_text,
            "resume": resume_text
        })
        return response.content
    except Exception as e:
        return f"Could not optimize resume: {e}"
