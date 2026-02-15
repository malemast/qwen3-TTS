# Qwen3-TTS Voice Cloning Configuration

## Working Setup (Feb 2026)

### Model
- **Model ID**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- **Why Base?**: Only the Base variant supports voice cloning. CustomVoice has better built-in speakers but no cloning.

### Infrastructure
- **Platform**: RunPod GPU Pod
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Image**: `ghcr.io/malemast/qwen3-tts:latest`
- **Port**: 8000/http
- **Proxy URL**: `https://{pod_id}-8000.proxy.runpod.net`

### Voice Cloning Settings
```json
{
  "text": "Your text to synthesize",
  "reference_audio_base64": "data:audio/wav;base64,{base64_encoded_wav}",
  "x_vector_only": true,
  "language": "English",
  "output_format": "wav"
}
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `x_vector_only` | `true` | **Critical**: Use x-vector mode. Does NOT require transcript. |
| `reference_audio_base64` | data URI | WAV file, base64 encoded with `data:audio/wav;base64,` prefix |
| `reference_text` | (optional) | Only needed if `x_vector_only=false` (ICL mode) |
| `language` | "English" | Supports: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |

### Voice Sample Requirements
- Format: WAV (recommended) or MP3
- Duration: 10-30 seconds ideal
- Quality: Clean audio, minimal background noise
- Content: Natural speech in the target language

### Working Voice Sample
- **File**: `claire_kore_25s.wav`
- **Duration**: ~25 seconds
- **Source**: Recorded sample

## API Endpoints

### Generate Speech
```bash
POST /generate
Content-Type: application/json

{
  "text": "Hello, how are you today?",
  "reference_audio_base64": "data:audio/wav;base64,...",
  "x_vector_only": true,
  "language": "English",
  "output_format": "wav"
}
```

### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

## Reproduction Steps

1. **Deploy to RunPod**:
   ```bash
   # Use RunPod API or console
   GPU: NVIDIA GeForce RTX 4090
   Image: ghcr.io/malemast/qwen3-tts:latest
   Ports: 8000/http
   Container Disk: 50GB
   ```

2. **Wait for model to load** (~60-90 seconds after pod starts)

3. **Test voice cloning**:
   ```python
   import base64
   import httpx

   # Load voice sample
   with open("voice_sample.wav", "rb") as f:
       voice_b64 = base64.b64encode(f.read()).decode()

   response = httpx.post(
       "https://{pod_id}-8000.proxy.runpod.net/generate",
       json={
           "text": "Your text here",
           "reference_audio_base64": f"data:audio/wav;base64,{voice_b64}",
           "x_vector_only": True,
           "language": "English",
           "output_format": "wav"
       },
       timeout=300.0
   )
   ```

## Commits That Fixed Voice Cloning

1. `41a1caf` - Upgrade to 1.7B model with pitch shift support
2. `7c4fc8b` - Add CORS middleware for browser testing
3. `4ff54e3` - Switch to Base model for voice cloning support
4. `1e834e4` - Make reference_text optional for voice cloning
5. `e0e7dae` - Add x_vector_only mode for voice cloning without transcript

## Troubleshooting

### "model does not support generate_voice_clone"
- You're using CustomVoice model. Switch to Base model.

### "ref_text is required when x_vector_only_mode=False"
- Set `x_vector_only: true` in your request, OR provide `reference_text`.

### Timeout errors
- Increase timeout to 300-600 seconds for longer text
- Text over ~500 words may need chunking
