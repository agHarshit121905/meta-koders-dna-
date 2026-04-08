FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Ensure src is importable
ENV PYTHONPATH="/app:/app/src"

# HF Spaces expects port 7860
EXPOSE 7860

USER appuser

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "."]
