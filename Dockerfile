# Use slim Python 3.11 on AMD64
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all code, models, metadata, and requirements file into the container
COPY . .

# Install system-level dependencies (for PyMuPDF and fonts if needed)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure input/output folders exist at runtime
RUN mkdir -p input output

# Default command to run your script
CMD ["python", "main.py"]
