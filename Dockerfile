FROM python:3.11.9-slim

# Install bash and standard tools
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Make the run script executable
RUN chmod +x run.sh

# Hugging Face Spaces requires exposing port 7860
EXPOSE 7860

# Point the Streamlit Frontend to the internal FastAPI Backend
ENV API_URL=http://localhost:8000

# Start both servers
CMD ["./run.sh"]
