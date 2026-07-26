FROM python:3.12-slim

WORKDIR /app

# System deps for healthcheck curl calls
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Install dependencies
COPY requirements.txt .

# sentence-transformers depends on torch, and a plain `pip install torch`
# on Linux pulls the full CUDA/GPU build (multiple GB of nvidia-*/cuda-toolkit
# packages we don't need in a CPU-only container). Installing the CPU-only
# wheel first satisfies that dependency before pip ever considers the GPU
# variant, and drastically cuts build time/image size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the local embedding model so it's baked into the image
# instead of downloading on first request. Only used when
# EMBEDDING_BACKEND=local (the default) -- see src/embedding_service.py.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY src /app/src

# Run as a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
