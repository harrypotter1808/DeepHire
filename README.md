# 🎯 HireSense AI: Professional Interview Preparation & Resume Optimization

HireSense AI is a high-performance, cloud-native platform designed to bridge the gap between candidates and their dream roles. Powered by the latest **Google Gemini 2.x** models and **LangChain**, it provides semantic resume matching, context-aware interview coaching, and automated ATS optimization.

## 🚀 Live Demo
*   **Frontend:** [https://deephire-frontend.onrender.com](https://deephire-frontend.onrender.com)
*   **Backend API:** [https://deephire-backend.onrender.com](https://deephire-backend.onrender.com)

## 🔥 Core Features
- **📊 A/B Comparison Dashboard:** Benchmarks TF-IDF baseline similarity against advanced semantic **BERT-embeddings**.
- **🎤 AI Interview Coach:** Engaging, context-aware technical and behavioral mock interviews using RAG (Retrieval-Augmented Generation).
- **🧠 Strategic AI Career Advisor:** Deep-context analysis that compares full resume history against a Job Description for real-world strategic pathing.
- **⚡ One-Click ATS Optimizer:** Instant resume rewriting using Gemini-authorized STAR methodology to resolve identified skill gaps.
- **🎨 LaTeX Professional Export:** One-click translation of optimized resumes into high-quality, professional LaTeX source code.
- **🤖 Autonomous AI Handshaking:** Dynamic discovery of the best available Gemini models (2.0/2.5) for 100% regional API compatibility.

## 🛠️ Technology Stack
*   **AI Engine:** Google Gemini 2.x, LangChain, FAISS (Vector Store)
*   **Backend:** FastAPI, Python 3.11, Uvicorn
*   **Frontend:** Streamlit, Requests
*   **Deployment:** Docker, Render Blueprints (Infrastructure as Code)

## 📁 Repository Structure
```text
├── backend/            # FastAPI Application & AI Services
│   ├── services/       # RAG, Matcher, and Parser logic
│   └── tests/          # Unit tests for scoring and parsing
├── frontend/           # Streamlit UI & Dashboard
│   └── pages/          # Multi-page application structure
├── .github/            # GitHub Actions CI/CD workflows
├── render.yaml         # Render Blueprint for Cloud Deployment
└── README.md           # Project Documentation
```

## ⚙️ Local Setup
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/harrypotter1808/DeepHire.git
    cd DeepHire
    ```
2.  **Environment Variables:**
    Create a `.env` file in the `backend/` directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```
3.  **Run with Docker:**
    ```bash
    docker-compose up --build
    ```

---
*Built with ❤️ for the future of recruitment.*
