"""
RunPod Serverless Handler for Qwen3-TTS.

Supports:
- Text-to-speech with voice cloning from reference audio
- Natural language style instructions
- Base64 audio input/output for voice samples
"""

import runpod
import base64
import io
import logging
import os

import torch
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded once at worker startup)
_model = None

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 24000


def load_model():
    """Load Qwen3-TTS model on startup."""
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


def generate_speech(
    text: str,
    reference_audio: np.ndarray | None = None,
    reference_audio_sr: int = SAMPLE_RATE,
    reference_text: str | None = None,
    language: str = "English",
    speaker: str = "Vivian",
    style_instruction: str | None = None,
) -> tuple[np.ndarray, float]:
    """
    Generate speech from text using Qwen3-TTS.

    Args:
        text: Text to synthesize
        reference_audio: Optional reference audio for voice cloning
        reference_audio_sr: Sample rate of reference audio
        reference_text: Transcript of reference audio (required for cloning)
        language: Language for synthesis
        speaker: Built-in speaker name (if not cloning)
        style_instruction: Natural language style instruction

    Returns:
        Tuple of (audio_array, duration_seconds)
    """
    load_model()

    if reference_audio is not None and reference_text:
        # Voice cloning mode
        logger.info(f"Generating with voice cloning, ref audio: {len(reference_audio)} samples")

        wavs, sr = _model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=(reference_audio, reference_audio_sr),
            ref_text=reference_text,
        )
    else:
        # Use built-in voice
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


def handler(job):
    """
    RunPod serverless handler.

    Input format:
    {
        "text": "Hello, how are you?",
        "reference_audio_base64": "data:audio/wav;base64,...",  # Optional
        "reference_text": "This is me speaking.",               # Required if using reference_audio
        "language": "English",                                  # Optional
        "speaker": "Vivian",                                    # Optional (if not cloning)
        "style_instruction": "Speak warmly",                    # Optional
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
    speaker = job_input.get("speaker", "Vivian")
    style_instruction = job_input.get("style_instruction")
    output_format = job_input.get("output_format", "wav")

    logger.info(f"Processing TTS request: {len(text)} chars, format={output_format}")

    # Process reference audio if provided
    reference_audio = None
    reference_audio_sr = SAMPLE_RATE
    if reference_audio_b64:
        if not reference_text:
            return {"error": "reference_text is required when using reference_audio_base64"}
        try:
            audio, sr = decode_audio_base64(reference_audio_b64)
            reference_audio = audio
            reference_audio_sr = sr
            # Limit to 15 seconds
            max_samples = 15 * sr
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
            logger.info(f"Reference audio: {len(reference_audio)/sr:.2f}s")
        except Exception as e:
            logger.error(f"Failed to decode reference audio: {e}")
            return {"error": f"Failed to decode reference audio: {str(e)}"}

    # Generate speech
    try:
        audio, duration = generate_speech(
            text=text,
            reference_audio=reference_audio,
            reference_audio_sr=reference_audio_sr,
            reference_text=reference_text,
            language=language,
            speaker=speaker,
            style_instruction=style_instruction,
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
