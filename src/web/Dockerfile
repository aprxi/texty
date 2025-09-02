FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary modules
COPY src/evaluator /app/src/evaluator
COPY src/helpers /app/src/helpers

# Copy the web application
COPY src/web /app/

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]