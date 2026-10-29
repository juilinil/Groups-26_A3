"""A's loading/resampling rule, reused by B's training and speaker extraction."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000


def load_audio_for_speecht5(audio_path):
    waveform, original_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Empty or non-finite audio: {audio_path}")
    waveform = waveform.mean(axis=1)
    if original_sr != TARGET_SAMPLE_RATE:
        divisor = math.gcd(original_sr, TARGET_SAMPLE_RATE)
        waveform = resample_poly(waveform, up=TARGET_SAMPLE_RATE // divisor,
                                 down=original_sr // divisor)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid processed audio: {audio_path}")
    return waveform, TARGET_SAMPLE_RATE
