# ═══════════════════════════════════════════════════
#  RAG Assistant v2 — Multi-stage Docker Build
# ═══════════════════════════════════════════════════

# ── Stage 1: Build React frontend ──
FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY package*.json ./
RUN npm install --no-audit --no-fund
COPY src/ ./src/
COPY public/ ./public/
RUN npm run build

# ── Stage 2: Python runtime ──
FROM python:3.12-slim AS runtime

# System deps for PyMuPDF, sentence-transformers, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py main.py ./
COPY core/ ./core/
COPY api/ ./api/

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/build ./frontend/build

# Create data directories
RUN mkdir -p /app/data/chroma_db /app/data/tree_indexes /app/data/pageindex_uploads /app/docs

# Non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]