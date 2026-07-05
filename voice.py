"""Voice input: record the microphone → transcribe to text with local Whisper (mlx-whisper).

An optional feature (triggered by typing 'v' in chat.py). Apple Silicon only (mlx-whisper).
- Recording uses sounddevice (cross-platform); transcription uses mlx-whisper (local,
  multilingual, offline, free).
- The sounddevice float32 array is fed straight to whisper → no file on disk, no ffmpeg needed.
- The transcribed text (in any language) is handed to chat.condense_question to be normalized
  into English before retrieval.

The model defaults to whisper-large-v3-turbo (multilingual; first run auto-downloads ~1.6G from HF).
For a quick test: export STT_MODEL=mlx-community/whisper-tiny (small, fast, but lower accuracy).
"""
import os

STT_MODEL = os.environ.get("STT_MODEL", "mlx-community/whisper-large-v3-turbo")
SAMPLE_RATE = 16000        # Whisper wants 16kHz
DEFAULT_SECONDS = 8


def record(seconds=DEFAULT_SECONDS):
    """Record the microphone for `seconds` → a float32 mono array (16kHz)."""
    import sounddevice as sd
    print(f"🎤 Recording {seconds}s, please speak…", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def transcribe(audio):
    """A float32 audio array (16kHz) → text. Fed directly as an array, no file, no ffmpeg."""
    import mlx_whisper
    return mlx_whisper.transcribe(audio, path_or_hf_repo=STT_MODEL)["text"].strip()


def listen(seconds=DEFAULT_SECONDS):
    """Record + transcribe → text (all in one step)."""
    return transcribe(record(seconds))
