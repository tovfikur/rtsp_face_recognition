# 1) BUILD STAGE: install dependencies and compile wheels
FROM python:3.9-slim AS builder

# Avoid .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS packages needed for builds, including CMake for dlib
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpng-dev \
        libjpeg-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install pip build tools to satisfy PEP 517 backends
RUN pip install --upgrade pip setuptools wheel build

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install  -r requirements.txt
RUN pip install "uvicorn[standard]"

# 2) RUNTIME STAGE: slim image for production or dev
FROM python:3.9-slim AS runtime

# Avoid .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime OS packages (OpenCV libs, ffmpeg, sudo)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpng-dev \
        libjpeg-dev \
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

# Create logs directory with appropriate permissions
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code into image
COPY . /app

# Allow mounting /app for live development
VOLUME ["/app"]

# Copy OpenFace model files (ensure these are in your project directory)
COPY shape_predictor_68_face_landmarks.dat /app/
COPY nn4.small2.v1.t7 /app/

# Copy entrypoint script and grant execution
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Entrypoint will choose between dev (--reload) and prod (gunicorn)
ENTRYPOINT ["/entrypoint.sh"]