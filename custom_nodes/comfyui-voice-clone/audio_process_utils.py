"""Audio preprocess/postprocess helpers aligned with OmniVoice reference handling."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence


def audio_fingerprint(audio: dict) -> str:
    waveform = audio["waveform"]
    if isinstance(waveform, torch.Tensor):
        flat = waveform.reshape(-1)
        numel = flat.numel()
        if numel == 0:
            samples = "empty"
        else:
            samples = f"{flat[0].item()}|{flat[numel // 2].item()}|{flat[-1].item()}"
    else:
        arr = np.asarray(waveform).reshape(-1)
        numel = arr.size
        if numel == 0:
            samples = "empty"
        else:
            samples = f"{arr[0]}|{arr[numel // 2]}|{arr[-1]}"
    return f"{audio['sample_rate']}|{tuple(waveform.shape)}|{numel}|{samples}"


def comfy_audio_to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    waveform = audio["waveform"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze(0).cpu().numpy()
    else:
        waveform = np.asarray(waveform).squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    return waveform.astype(np.float32, copy=False), int(audio["sample_rate"])


def numpy_to_comfy_audio(waveform: np.ndarray, sample_rate: int) -> dict:
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    tensor = torch.from_numpy(waveform.astype(np.float32, copy=False)).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    tensor = torch.from_numpy(wav)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.numpy()


def _numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def _audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def _remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int,
    trail_sil: int,
    silence_threshold: float = -50,
) -> AudioSegment:
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    return audio.reverse()


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    wave = _numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        if non_silent_segs:
            wave = sum(non_silent_segs[1:], non_silent_segs[0])
        else:
            wave = AudioSegment.silent(duration=0)

    wave = _remove_silence_edges(wave, lead_sil, trail_sil, -50)
    return _audiosegment_to_numpy(wave)


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in
            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros((processed.shape[0], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def trim_to_max_duration(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float,
) -> np.ndarray:
    if max_duration <= 0:
        return audio
    max_samples = int(max_duration * sampling_rate)
    if audio.shape[-1] <= max_samples:
        return audio
    return audio[:, :max_samples]


def preprocess_reference_audio(
    audio: dict,
    target_sample_rate: int = 24000,
    max_duration: float = 10.0,
    remove_silence_enabled: bool = True,
    mid_sil_ms: int = 200,
    lead_sil_ms: int = 100,
    trail_sil_ms: int = 200,
    normalize_quiet: bool = True,
    quiet_rms_threshold: float = 0.1,
) -> tuple[dict, float]:
    wav, sr = comfy_audio_to_numpy(audio)

    if wav.shape[0] > 1:
        wav = np.mean(wav, axis=0, keepdims=True)

    # Trim first so resample/silence removal only run on the kept segment.
    wav = trim_to_max_duration(wav, sr, max_duration)

    if sr != target_sample_rate:
        wav = _resample(wav, sr, target_sample_rate)
        sr = target_sample_rate

    ref_rms = float(np.sqrt(np.mean(wav**2)))
    if normalize_quiet and 0 < ref_rms < quiet_rms_threshold:
        wav = wav * quiet_rms_threshold / ref_rms

    if remove_silence_enabled:
        wav = remove_silence(
            wav,
            sr,
            mid_sil=mid_sil_ms,
            lead_sil=lead_sil_ms,
            trail_sil=trail_sil_ms,
        )
        if wav.shape[-1] == 0:
            raise ValueError(
                "Reference audio is empty after silence removal. "
                "Disable remove_silence or use different source audio."
            )

    return numpy_to_comfy_audio(wav, sr), ref_rms


def postprocess_generated_audio(
    audio: dict,
    ref_rms: float = 0.0,
    remove_silence_enabled: bool = True,
    mid_sil_ms: int = 500,
    lead_sil_ms: int = 100,
    trail_sil_ms: int = 100,
    fade_duration: float = 0.1,
    pad_duration: float = 0.1,
    quiet_rms_threshold: float = 0.1,
    peak_normalize_level: float = 0.5,
) -> dict:
    wav, sr = comfy_audio_to_numpy(audio)

    if remove_silence_enabled:
        wav = remove_silence(
            wav,
            sr,
            mid_sil=mid_sil_ms,
            lead_sil=lead_sil_ms,
            trail_sil=trail_sil_ms,
        )

    if ref_rms > 0 and ref_rms < quiet_rms_threshold:
        wav = wav * ref_rms / quiet_rms_threshold
    elif ref_rms <= 0:
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * peak_normalize_level

    wav = fade_and_pad_audio(
        wav,
        pad_duration=pad_duration,
        fade_duration=fade_duration,
        sample_rate=sr,
    )
    return numpy_to_comfy_audio(wav, sr)
