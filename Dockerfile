FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8001

# Health check (start-period long pour charger 74k articles + modèle d'AlemBERT)
HEALTHCHECK --interval=30s --timeout=60s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

# Run the HTTP server
CMD ["python", "-m", "encycloscope_mcp.http_streamable"]
