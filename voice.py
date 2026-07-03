"""语音输入：录麦克风 → 本地 Whisper(mlx-whisper) 转文字。

可选功能（chat.py 里输入 'v' 触发）。仅 Apple Silicon（mlx-whisper）。
- 录音用 sounddevice（跨平台），转写用 mlx-whisper（本地、多语言、离线、免费）。
- 直接把 sounddevice 的 float32 数组喂给 whisper → 不落文件、不需要 ffmpeg。
- 转出的文字（任何语言）交给 chat.condense_question 规整成英文再检索。

模型默认 whisper-large-v3-turbo（多语言，首次自动从 HF 下载 ~1.6G）。
想快速试：export STT_MODEL=mlx-community/whisper-tiny（小、快、但准度低）。
"""
import os

STT_MODEL = os.environ.get("STT_MODEL", "mlx-community/whisper-large-v3-turbo")
SAMPLE_RATE = 16000        # Whisper 要 16kHz
DEFAULT_SECONDS = 8


def record(seconds=DEFAULT_SECONDS):
    """录 seconds 秒麦克风 → float32 单声道数组(16kHz)。"""
    import sounddevice as sd
    print(f"🎤 录音 {seconds}s，请说话…", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def transcribe(audio):
    """float32 音频数组(16kHz) → 文字。直接喂数组，不经文件、不需要 ffmpeg。"""
    import mlx_whisper
    return mlx_whisper.transcribe(audio, path_or_hf_repo=STT_MODEL)["text"].strip()


def listen(seconds=DEFAULT_SECONDS):
    """录音 + 转写 → 文字（一步到位）。"""
    return transcribe(record(seconds))
