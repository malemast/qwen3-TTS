"""
RunPod Serverless Handler for Qwen3-TTS.

Supports:
- Text-to-speech with voice cloning from reference audio
- Natural language emotion tags (Qwen3's strength)
- Base64 audio input/output for voice samples
"""

import runpod
import base64
import io
import logging
import tempfile
import os
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded once at worker startup)
_model = None
_processor = None

MODEL_ID = "Qwen/Qwen3-TTS"
SAMPLE_RATE = 24000


def load_model():
    """Load Qwen3-TTS model on startup."""
    global _model, _processor

    if _model is not None:
        return

    logger.info(f"Loading Qwen3-TTS model: {MODEL_ID}")

    from transformers import AutoProcessor, Qwen3TTSForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    logger.info("Qwen3-TTS model loaded successfully")


def decode_audio_base64(audio_b64: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio to numpy array."""
    # Handle data URI format
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
        # Simple resampling fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def generate_speech(
    text: str,
    reference_audio: np.ndarray | None = None,
    reference_text: str | None = None,
    language: str = "English",
) -> tuple[np.ndarray, float]:
    """
    Generate speech from text using Qwen3-TTS.

    Args:
        text: Text to synthesize (can include emotion tags like [happy], [sad])
        reference_audio: Optional reference audio for voice cloning (24kHz)
        reference_text: Transcript of reference audio (improves cloning quality)
        language: Language for synthesis

    Returns:
        Tuple of (audio_array, duration_seconds)
    """
    load_model()

    # Build the prompt
    if reference_audio is not None:
        # Voice cloning mode
        logger.info(f"Generating with voice cloning, ref audio: {len(reference_audio)} samples")

        # Qwen3-TTS expects reference audio and optional transcript
        inputs = _processor(
            text=text,
            audio=reference_audio,
            audio_transcript=reference_text,
            return_tensors="pt",
        )
    else:
        # Default voice mode
        logger.info("Generating with default voice")
        inputs = _processor(
            text=text,
            return_tensors="pt",
        )

    # Move to GPU
    inputs = {k: v.to(_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=4096,
        )

    # Decode audio
    audio = _processor.decode(outputs[0])
    audio_np = audio.cpu().numpy().squeeze()

    duration = len(audio_np) / SAMPLE_RATE
    logger.info(f"Generated {duration:.2f}s of audio")

    return audio_np, duration


def handler(job):
    """
    RunPod serverless handler.

    Input format:
    {
        "text": "Hello, how are you?",
        "reference_audio_base64": "data:audio/wav;base64,...",  # Optional
        "reference_text": "This is me speaking.",               # Optional
        "language": "English",                                  # Optional
        "output_format": "wav"                                  # Optional: wav or mp3
    }

    Output format:
    {
        "audio_base64": "data:audio/wav;base64,...",
        "duration_seconds": 2.5,
        "sample_rate": 24000
    }
    """
    job_input = job["input"]

    # Extract parameters
    text = job_input.get("text")
    if not text:
        return {"error": "Missing required 'text' parameter"}

    reference_audio_b64 = job_input.get("reference_audio_base64")
    reference_text = job_input.get("reference_text")
    language = job_input.get("language", "English")
    output_format = job_input.get("output_format", "wav")

    logger.info(f"Processing TTS request: {len(text)} chars, format={output_format}")

    # Process reference audio if provided
    reference_audio = None
    if reference_audio_b64:
        try:
            audio, sr = decode_audio_base64(reference_audio_b64)
            # Resample to 24kHz if needed
            reference_audio = resample_audio(audio, sr, SAMPLE_RATE)
            # Limit to 15 seconds
            max_samples = 15 * SAMPLE_RATE
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
            logger.info(f"Reference audio: {len(reference_audio)/SAMPLE_RATE:.2f}s")
        except Exception as e:
            logger.error(f"Failed to decode reference audio: {e}")
            return {"error": f"Failed to decode reference audio: {str(e)}"}

    # Generate speech
    try:
        audio, duration = generate_speech(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            language=language,
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return {"error": f"TTS generation failed: {str(e)}"}

    # Encode output
    try:
        audio_b64 = encode_audio_base64(audio, SAMPLE_RATE, format=output_format)
    except Exception as e:
        logger.error(f"Failed to encode audio: {e}")
        return {"error": f"Failed to encode audio: {str(e)}"}

    return {
        "audio_base64": audio_b64,
        "duration_seconds": duration,
        "sample_rate": SAMPLE_RATE,
    }


# Load model at startup (cold start optimization)
if os.environ.get("RUNPOD_POD_ID"):
    logger.info("Running on RunPod, loading model...")
    load_model()


runpod.serverless.start({"handler": handler})
