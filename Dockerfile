# Multi-stage build for production
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.prod.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.prod.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    NODE_ENV=production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && update-ca-certificates

# Create non-root user with specific UID/GID for security
RUN groupadd -r appuser -g 1000 && \
    useradd -r -u 1000 -g appuser -s /bin/bash -m appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/cache /app/logs /app/temp /app/backups /app/Results \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/logs /app/temp

# Create SSL directory for certificates
RUN mkdir -p /etc/ssl/certs /etc/ssl/private \
    && chown -R appuser:appuser /etc/ssl

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8003 8501 8502

# Health check with improved configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8003/health || exit 1

# Security: Set read-only root filesystem (except for writable directories)
VOLUME ["/app/data", "/app/cache", "/app/logs", "/app/temp", "/app/backups", "/app/Results"]

# Default command with proper signal handling
CMD ["python", "main.py"]
