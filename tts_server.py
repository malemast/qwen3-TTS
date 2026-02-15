"""
Simple HTTP TTS Server for Qwen3-TTS.

Runs on the RunPod GPU pod and serves TTS requests via HTTP.
This is used instead of serverless for batch processing.
"""

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_model = None

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SAMPLE_RATE = 24000


class TTSRequest(BaseModel):
    """TTS generation request."""
    text: str
    reference_audio_base64: Optional[str] = None
    reference_text: Optional[str] = None
    language: str = "English"
    output_format: str = "wav"  # wav or mp3
    speaker: str = "Vivian"  # Default speaker for non-cloned voice
    style_instruction: Optional[str] = None  # e.g., "Speak warmly and naturally"
    pitch_shift: float = 0.0  # Semitones to shift pitch (-2 = deeper, +2 = higher)
    x_vector_only: bool = True  # If True, use x-vector mode (no transcript needed). If False, use ICL mode (needs transcript)


class TTSResponse(BaseModel):
    """TTS generation response."""
    audio_base64: str
    duration_seconds: float
    sample_rate: int = SAMPLE_RATE


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None


def load_model():
    """Load Qwen3-TTS model."""
    global _model

    if _model is not None:
        return

    logger.info(f"Loading Qwen3-TTS model: {MODEL_ID}")

    from qwen_tts import Qwen3TTSModel

    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    logger.info("Qwen3-TTS model loaded successfully")


def decode_audio_base64(audio_b64: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio to numpy array."""
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",")[1]

    audio_bytes = base64.b64decode(audio_b64)
    audio_buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(audio_buffer)

    return audio.astype(np.float32), sr


def encode_audio_base64(audio: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """Encode numpy audio to base64."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=format)
    audio_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    mime_type = "audio/wav" if format == "wav" else "audio/mpeg"
    return f"data:{mime_type};base64,{audio_b64}"


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def generate_speech(
    text: str,
    reference_audio: Optional[np.ndarray] = None,
    reference_audio_sr: int = SAMPLE_RATE,
    reference_text: Optional[str] = None,
    language: str = "English",
    speaker: str = "Vivian",
    style_instruction: Optional[str] = None,
    x_vector_only: bool = True,
) -> tuple[np.ndarray, float]:
    """Generate speech from text."""
    load_model()

    if reference_audio is not None:
        # Voice cloning mode
        mode = "x-vector" if x_vector_only else "ICL"
        logger.info(f"Generating with voice cloning ({mode} mode), ref: {len(reference_audio)} samples")

        clone_kwargs = {
            "text": text,
            "language": language,
            "ref_audio": (reference_audio, reference_audio_sr),
            "x_vector_only_mode": x_vector_only,
        }
        # Only include ref_text if provided (required for ICL mode)
        if reference_text:
            clone_kwargs["ref_text"] = reference_text

        wavs, sr = _model.generate_voice_clone(**clone_kwargs)
    else:
        # Use built-in voice with optional style instruction
        logger.info(f"Generating with speaker: {speaker}")

        instruct = style_instruction or "Speak naturally and clearly"

        wavs, sr = _model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

    audio_np = wavs[0] if isinstance(wavs, list) else wavs
    if hasattr(audio_np, 'cpu'):
        audio_np = audio_np.cpu().numpy()

    duration = len(audio_np) / sr
    logger.info(f"Generated {duration:.2f}s of audio")

    return audio_np, duration


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting TTS server...")
    load_model()
    yield
    logger.info("Shutting down TTS server...")


app = FastAPI(
    title="Qwen3-TTS Server",
    description="Text-to-speech with voice cloning",
    lifespan=lifespan
)

# Enable CORS for browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


@app.post("/generate", response_model=TTSResponse)
async def generate(request: TTSRequest):
    """Generate speech from text."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Process reference audio if provided
    reference_audio = None
    reference_audio_sr = SAMPLE_RATE
    if request.reference_audio_base64:
        try:
            audio, sr = decode_audio_base64(request.reference_audio_base64)
            reference_audio = audio
            reference_audio_sr = sr
            # Limit to 15 seconds
            max_samples = 15 * sr
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode reference audio: {str(e)}"
            )

    # Generate in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        audio, duration = await loop.run_in_executor(
            None,
            lambda: generate_speech(
                text=request.text,
                reference_audio=reference_audio,
                reference_audio_sr=reference_audio_sr,
                reference_text=request.reference_text,
                language=request.language,
                speaker=request.speaker,
                style_instruction=request.style_instruction,
                x_vector_only=request.x_vector_only,
            )
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    # Apply pitch shift if requested (negative = deeper, positive = higher)
    if request.pitch_shift != 0.0:
        try:
            import librosa
            logger.info(f"Applying pitch shift: {request.pitch_shift} semitones")
            audio = librosa.effects.pitch_shift(
                audio,
                sr=SAMPLE_RATE,
                n_steps=request.pitch_shift
            )
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")

    # Encode output
    audio_b64 = encode_audio_base64(audio, SAMPLE_RATE, format=request.output_format)

    return TTSResponse(
        audio_base64=audio_b64,
        duration_seconds=duration,
        sample_rate=SAMPLE_RATE,
    )


@app.post("/generate_and_save")
async def generate_and_save(request: TTSRequest):
    """Generate speech and return raw audio bytes (for large files)."""
    result = await generate(request)

    # Strip data URI prefix
    audio_b64 = result.audio_base64
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",")[1]

    return {
        "audio_bytes": audio_b64,
        "duration_seconds": result.duration_seconds,
        "sample_rate": result.sample_rate,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
