# ==============================================================================
# CardioQ: Production Multi-Stage Container Image
# Precision Cardiovascular Intelligence Platform
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set non-interactive debian frontend and python optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    HOST=0.0.0.0

# Install minimal system dependencies for C-extensions, OpenMP, and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for Docker caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code, models, datasets, and configurations
COPY . .

# Create directory structure for runtime uploads and jobs
RUN mkdir -p artifacts/uploads artifacts/training_jobs artifacts/models

# Expose standard web port
EXPOSE 8080

# Health check to ensure zero-downtime rolling deploys
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Launch the clinical platform server
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]
