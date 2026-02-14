# RunPod GPU Pod for Qwen3-TTS
# Runs HTTP server for batch TTS processing

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and uvicorn for HTTP server
RUN pip install --no-cache-dir fastapi uvicorn

# Pre-download the model during build (reduces startup time)
RUN python -c "from transformers import AutoProcessor; \
    AutoProcessor.from_pretrained('Qwen/Qwen3-TTS', trust_remote_code=True); \
    print('Processor downloaded')"

RUN python -c "from transformers import Qwen3TTSForConditionalGeneration; \
    Qwen3TTSForConditionalGeneration.from_pretrained('Qwen/Qwen3-TTS', trust_remote_code=True); \
    print('Model downloaded')"

# Copy server code
COPY tts_server.py .
COPY handler.py .

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run HTTP server (for GPU Pods)
# For serverless, override with: python handler.py
CMD ["python", "-u", "tts_server.py"]
