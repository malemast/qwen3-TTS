#!/usr/bin/env python3
"""
Local test script for Qwen3-TTS handler.

Run without RunPod to test the TTS functionality.
"""

import sys
import base64
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from handler import generate_speech, decode_audio_base64, encode_audio_base64, SAMPLE_RATE
import soundfile as sf
import numpy as np


def test_basic_generation():
    """Test basic TTS without voice cloning."""
    print("Testing basic generation...")

    text = "Hello! This is a test of Qwen3 text to speech. [happy] Isn't this exciting?"

    audio, duration = generate_speech(text)

    output_path = Path("test_output_basic.wav")
    sf.write(output_path, audio, SAMPLE_RATE)

    print(f"  Generated: {duration:.2f}s")
    print(f"  Saved to: {output_path}")
    return True


def test_voice_cloning(reference_path: str):
    """Test voice cloning with a reference sample."""
    print(f"Testing voice cloning with: {reference_path}")

    # Load reference audio
    ref_audio, ref_sr = sf.read(reference_path)
    if ref_sr != SAMPLE_RATE:
        print(f"  Resampling from {ref_sr}Hz to {SAMPLE_RATE}Hz...")
        try:
            import librosa
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=SAMPLE_RATE)
        except ImportError:
            print("  Warning: librosa not available, using raw audio")

    # Limit to 15 seconds
    max_samples = 15 * SAMPLE_RATE
    if len(ref_audio) > max_samples:
        ref_audio = ref_audio[:max_samples]
        print(f"  Trimmed to 15 seconds")

    text = "[excited] Oh my goodness, this voice cloning is incredible! I sound just like the reference."

    audio, duration = generate_speech(
        text=text,
        reference_audio=ref_audio.astype(np.float32),
        reference_text="This is a sample of my voice speaking naturally.",
    )

    output_path = Path("test_output_cloned.wav")
    sf.write(output_path, audio, SAMPLE_RATE)

    print(f"  Generated: {duration:.2f}s")
    print(f"  Saved to: {output_path}")
    return True


def test_emotion_tags():
    """Test different emotion tags."""
    print("Testing emotion tags...")

    emotions = [
        ("[happy] This makes me so happy!", "happy"),
        ("[sad] I'm feeling a bit down today.", "sad"),
        ("[excited] We won! We actually won!", "excited"),
        ("[whisper] Don't tell anyone this secret.", "whisper"),
    ]

    for text, emotion in emotions:
        print(f"  Generating: {emotion}...")
        audio, duration = generate_speech(text)

        output_path = Path(f"test_output_{emotion}.wav")
        sf.write(output_path, audio, SAMPLE_RATE)
        print(f"    {duration:.2f}s -> {output_path}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen3-TTS locally")
    parser.add_argument(
        "--reference", "-r",
        help="Path to reference audio for voice cloning test"
    )
    parser.add_argument(
        "--basic", "-b",
        action="store_true",
        help="Run basic generation test"
    )
    parser.add_argument(
        "--emotions", "-e",
        action="store_true",
        help="Run emotion tags test"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all tests"
    )

    args = parser.parse_args()

    if args.all or args.basic:
        test_basic_generation()
        print()

    if args.all or args.emotions:
        test_emotion_tags()
        print()

    if args.reference:
        test_voice_cloning(args.reference)
        print()

    if not (args.all or args.basic or args.emotions or args.reference):
        print("No tests specified. Use --help for options.")
        print("Running basic test by default...")
        test_basic_generation()
