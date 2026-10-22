FROM python:3.12.9-slim

# Unprivileged user that runs the app
RUN useradd --create-home --uid 1000 app

WORKDIR /app

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Embedding model cached in the app user's home
ENV HF_HOME=/home/app/.cache/huggingface
USER app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
ENV HF_HUB_OFFLINE=1

# Source stays root-owned (read-only for the app); only data/ is writable
USER root
COPY . .
RUN mkdir -p /app/data/refined && chown -R app:app /app/data

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
