FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Pre-download the sentence transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY optimized_persona_analyzer.py .
COPY run_analyzer.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the default command
CMD ["python", "run_analyzer.py"]
