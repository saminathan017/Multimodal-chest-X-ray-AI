# ─────────────────────────────────────────────────────────────────
# Dockerfile — ClinicalAI Multi-Modal CDS
# Build:  docker build -t clinical-ai-demo .
# Run:    docker run -p 8501:8501 --env-file .env clinical-ai-demo
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Streamlit config
RUN mkdir -p ~/.streamlit && echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
