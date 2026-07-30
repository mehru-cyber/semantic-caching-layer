FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Same fix as the api service's Dockerfile: Railway assigns a dynamic $PORT
# and healthchecks against it, so the app must actually listen there
# instead of a hardcoded port. Falls back to 8501 locally (docker-compose),
# where $PORT is unset.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# Shell form (not exec/JSON-array form) so $PORT actually gets expanded by
# the shell at container start.
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
