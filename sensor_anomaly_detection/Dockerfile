# Start from a lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Environment variables for production
ENV PYTHONUNBUFFERED=1
ENV MODEL_CONSENSUS_THRESHOLD=1.0

CMD ["python", "sensor_clustering.py"]
