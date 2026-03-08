FROM python:3.11-slim

# Install system dependencies required by faiss-cpu and umap-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
# If requirements.txt doesn't change, this layer is reused on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create necessary directories (gitkeep files may not persist in image)
RUN mkdir -p data/artifacts data/metadata logs analysis/results

# Expose port 8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
