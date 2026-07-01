FROM python:3.11-slim

# System deps for pdfplumber / poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw_pdfs data/processed data/vector_db logs

# NOTE: Ollama must run as a separate service.
# See docker-compose.yml — the 'ollama' service handles this.

EXPOSE 8000 8501
