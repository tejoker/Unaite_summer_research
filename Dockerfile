# Multi-stage build for Paul Wurth Industrial Time Series Analysis Platform
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as cuda-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

FROM cuda-base as cpu-base
# CPU-only variant for systems without GPU

FROM cuda-base as production

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY executable/ ./executable/
COPY *.py ./
COPY README.md ./

# Create data and results directories
RUN mkdir -p data/{Golden,Anomaly,Test,Reconstruction} \
    && mkdir -p results logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4

# Default configuration
ENV PAUL_WURTH_CONFIG_PATH=/app/config
ENV PAUL_WURTH_DATA_PATH=/app/data
ENV PAUL_WURTH_RESULTS_PATH=/app/results

# Create non-root user for security
RUN useradd -m -u 1000 pwuser && chown -R pwuser:pwuser /app
USER pwuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import pandas; print('OK')" || exit 1

# Default command - show help
CMD ["python3", "executable/launcher.py", "--help"]

# Build variants:
# docker build -t paulwurth:cuda .
# docker build -t paulwurth:cpu --target cpu-base .