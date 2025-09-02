FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (for sentence tokenization)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/train /data/test /data/models /data/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/data/models/model.pkl
ENV LOG_LEVEL=INFO

# Default command
CMD ["python", "--version"]