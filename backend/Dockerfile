FROM python:3.10

WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install CPU-only PyTorch (much smaller)
RUN pip install --upgrade pip --no-cache-dir && \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir && \
    pip install --no-cache-dir -r requirements-prod.txt

# Clean up pip cache
RUN rm -rf /root/.cache/pip

# Verify installations
RUN python -c "import torch; print('PyTorch:', torch.__version__)" && \
    python -c "import faiss; print('Faiss:', faiss.__version__)"

# Copy app code
COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]