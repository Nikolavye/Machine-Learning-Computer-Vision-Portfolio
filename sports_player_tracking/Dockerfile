# Start from a CUDA-enabled PyTorch image for potential GPU acceleration, 
# or use CPU-only if strictly deploying to edge/Lambda.
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics opencv-python-headless

# Copy application files
COPY . .

# Expose port if serving via FastAPI (e.g., 8080)
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the tracking pipeline (or a FastAPI server in reality)
CMD ["python", "solution.py"]
