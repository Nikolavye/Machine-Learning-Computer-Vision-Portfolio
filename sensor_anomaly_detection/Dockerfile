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

# Environment variables for production
ENV PYTHONUNBUFFERED=1
ENV MODEL_CONSENSUS_THRESHOLD=1.0

# In a real batch pipeline (e.g., triggered by Airflow), we execute the main script.
CMD ["python", "sensor_clustering.py"]
