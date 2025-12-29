FROM python:3.10-slim

WORKDIR /app

# System deps (safe for RAG)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend ./backend

# Expose Hugging Face port
EXPOSE 7860

# Start Flask app from backend folder
CMD ["gunicorn", "-b", "0.0.0.0:7860", "backend.app:app"]
