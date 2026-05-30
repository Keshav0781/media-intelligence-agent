# Base image — Python 3.11 slim
# Using 3.11 not 3.14 — more stable, better ML library compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# FFmpeg — required for audio extraction from video
# build-essential — required for some Python packages to compile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user — security best practice
# Never run production containers as root
RUN useradd -m -u 1000 appuser

# Copy requirements first — Docker layer caching
# If requirements.txt unchanged, this layer is cached
# Only reinstalls packages when requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.yaml .
COPY conftest.py .
COPY pytest.ini .

# Create necessary directories and set ownership
RUN mkdir -p data/audio data/keyframes data/cache data/qdrant data/bm25 logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Streamlit port
EXPOSE 8501

# Health check — verifies config loads correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from app.config import get_config; get_config()" || exit 1

# Default command — run Streamlit UI
CMD ["streamlit", "run", "app/ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]