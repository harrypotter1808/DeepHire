#!/bin/bash

# Start the FastAPI backend on port 8000 in the background
echo "Starting FastAPI Backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to start
sleep 5

# Start the Streamlit frontend on Hugging Face's default port 7860
echo "Starting Streamlit Frontend..."
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
