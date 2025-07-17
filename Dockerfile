# 1) BUILD STAGE: install dependencies and compile wheels
FROM python:3.9-slim AS builder
# avoid .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Install OS packages needed for builds and OpenCV at once
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*
# Install pip build tools to satisfy PEP 517 backends
RUN pip install --upgrade pip setuptools wheel build
# Copy and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]"

# 2) RUNTIME STAGE: slim image for production or dev
FROM python:3.9-slim AS runtime
# avoid .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Install runtime OS packages (OpenCV libs, ffmpeg, sudo)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      ffmpeg \
      sudo && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with sudo privileges
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    echo "appuser:password123" | chpasswd && \
    usermod -aG sudo appuser

# Configure sudo to allow password authentication
RUN echo "appuser ALL=(ALL:ALL) ALL" >> /etc/sudoers

WORKDIR /app
# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy application code into image
COPY . /app
# Allow mounting /app for live development
VOLUME ["/app"]
# Copy entrypoint script and grant execution
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# Switch to non-root user
USER appuser
# Expose FastAPI port
EXPOSE 8000
# Entrypoint will choose between dev (--reload) and prod (gunicorn)
ENTRYPOINT ["/entrypoint.sh"]
