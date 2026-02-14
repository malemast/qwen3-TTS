# Qwen3-TTS on RunPod

High-quality text-to-speech with voice cloning, running on RunPod GPU pods.

## Features

- **Qwen3-TTS Model**: 0.6B parameter TTS model with excellent quality
- **Voice Cloning**: Clone any voice from a 5-15 second reference sample
- **Emotion Tags**: Natural language emotion control (`[happy]`, `[sad]`, `[excited]`, etc.)
- **HTTP API**: FastAPI server for easy integration
- **RunPod Ready**: Pre-built for RunPod GPU pods

## Quick Start

### Using the Pre-built Image

The Docker image is automatically built and pushed to GitHub Container Registry:

```bash
ghcr.io/malemast/qwen3-tts:latest
```

### Deploy on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Create a new GPU Pod
3. Select a GPU with 8GB+ VRAM (RTX 3070 minimum)
4. Use image: `ghcr.io/malemast/qwen3-tts:latest`
5. Expose HTTP port: `8000`
6. Start the pod

### API Usage

Once the pod is running:

```bash
# Health check
curl http://<pod-ip>:8000/health

# Generate speech
curl -X POST http://<pod-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "[happy] Hello, this is a test!"}'
```

## API Endpoints

### `GET /health`

Health check and status.

```json
{
    "status": "healthy",
    "model_loaded": true,
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3070"
}
```

### `POST /generate`

Generate speech from text.

**Request:**
```json
{
    "text": "[happy] Hello world!",
    "reference_audio_base64": "data:audio/wav;base64,...",
    "reference_text": "Optional transcript of reference",
    "language": "English",
    "output_format": "wav"
}
```

**Response:**
```json
{
    "audio_base64": "data:audio/wav;base64,...",
    "duration_seconds": 2.5,
    "sample_rate": 24000
}
```

## Emotion Tags

Qwen3-TTS supports natural language emotion tags:

```
[happy] Great to see you!
[sad] I'm sorry to hear that.
[excited] We won the game!
[whisper] Don't tell anyone, but...
[angry] I can't believe you did that!
[calm] Take a deep breath.
[laughing] That's hilarious!
```

## Voice Cloning

To clone a voice, provide:
1. `reference_audio_base64`: 5-15 seconds of the target voice (WAV format, base64 encoded)
2. `reference_text` (optional): Transcript of what's being said in the reference audio

Example:
```python
import base64
import requests

# Load reference audio
with open("voice_sample.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://<pod-ip>:8000/generate", json={
    "text": "[excited] This is voice cloning in action!",
    "reference_audio_base64": f"data:audio/wav;base64,{audio_b64}",
    "reference_text": "Hello, this is my natural speaking voice."
})

# Save output
audio_data = base64.b64decode(response.json()["audio_base64"].split(",")[1])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## GPU Requirements

| GPU | VRAM | $/hr | Speed | Notes |
|-----|------|------|-------|-------|
| RTX 3070 | 8GB | ~$0.10 | 2x RT | **Minimum** - tight fit but works |
| RTX 3080 | 10GB | ~$0.14 | 2.5x RT | Recommended |
| RTX 3090 | 24GB | ~$0.20 | 3x RT | Overkill for 0.6B model |
| RTX 4090 | 24GB | ~$0.34 | 4x RT | Maximum speed |

*RT = realtime (e.g., 2x RT means 1 second of audio takes 0.5 seconds to generate)*

## Building Locally

```bash
# Clone the repo
git clone https://github.com/malemast/qwen3-tts.git
cd qwen3-tts

# Build
docker build -t qwen3-tts:latest .

# Run locally (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 qwen3-tts:latest
```

## Local Testing (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server (requires CUDA GPU)
python tts_server.py

# Or run tests
python test_local.py --basic
python test_local.py --emotions
python test_local.py --reference voice_sample.wav
```

## License

MIT
