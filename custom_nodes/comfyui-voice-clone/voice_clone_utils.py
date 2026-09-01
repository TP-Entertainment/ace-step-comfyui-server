from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import requests
import torch
import torchaudio
from mutagen import File as MutagenFile

REF_AUDIO_MAX_SECONDS = 30.0
REF_AUDIO_CACHE_DIR = Path("temp/voice_ref")
REF_AUDIO_TRIM_STORAGE_FOLDER = "voice_ref_trim"


def hash_string(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def download_file(file_url: str, save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=2048 * 1024):
                if chunk:
                    f.write(chunk)

    return path


def _ffprobe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _audio_duration_seconds(path: Path) -> float:
    meta = MutagenFile(path)
    if meta is not None and meta.info is not None and getattr(meta.info, "length", None):
        return float(meta.info.length)
    return _ffprobe_duration_seconds(path)


def _trim_ref_text(ref_text: str, duration_sec: float, max_sec: float) -> str:
    if duration_sec <= max_sec or not ref_text.strip():
        return ref_text
    ratio = max_sec / duration_sec
    target_len = max(1, int(len(ref_text) * ratio))
    if target_len >= len(ref_text):
        return ref_text
    trimmed = ref_text[:target_len].rsplit(" ", 1)[0].strip()
    return trimmed or ref_text[:target_len].strip()


def _trim_audio_to_wav(source_path: Path, dest_path: Path, max_seconds: float) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-t",
            str(max_seconds),
            str(dest_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr.strip()}")


def _upload_to_storage(file_path: Path, storage_folder: str, storage_file_name: str) -> str:
    from google.cloud import storage
    from google.oauth2 import service_account

    storage_key_local_path = os.getenv("STORAGE_KEY_LOCAL_PATH")
    storage_key_url = os.getenv("STORAGE_KEY_URL")
    storage_url = os.getenv("STORAGE_URL")
    bucket_name = os.getenv("BUCKET")
    blob_prefix = os.getenv("BLOB")

    if not all([storage_key_local_path, storage_key_url, storage_url, bucket_name, blob_prefix]):
        raise RuntimeError(
            "Ref audio dài hơn 30s cần upload sau khi trim. "
            "Thiết lập STORAGE_KEY_LOCAL_PATH, STORAGE_KEY_URL, STORAGE_URL, BUCKET, BLOB."
        )

    key_path = Path(storage_key_local_path)
    if not key_path.exists():
        download_file(storage_key_url, key_path)

    credentials_info = json.loads(key_path.read_text(encoding="utf-8"))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    file_ext = file_path.suffix.lstrip(".")
    blob_name = f"{blob_prefix}/{storage_folder}/{storage_file_name}.{file_ext}"
    bucket.blob(blob_name).upload_from_filename(str(file_path))

    return f"{storage_url}/{bucket_name}/{blob_prefix}/{storage_folder}/{storage_file_name}.{file_ext}"


def _prepare_ref_audio_url(ref_audio_url: str) -> tuple[str, float]:
    REF_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hash_string(ref_audio_url)
    suffix = Path(ref_audio_url.split("?", 1)[0]).suffix or ".audio"
    source_path = REF_AUDIO_CACHE_DIR / f"ref_src_{url_hash}{suffix}"

    if not source_path.exists():
        download_file(ref_audio_url, source_path)

    duration_sec = _audio_duration_seconds(source_path)
    if duration_sec <= REF_AUDIO_MAX_SECONDS:
        return ref_audio_url, duration_sec

    trimmed_path = REF_AUDIO_CACHE_DIR / f"ref_trim_{url_hash}.wav"
    url_cache_path = REF_AUDIO_CACHE_DIR / f"ref_trim_{url_hash}.url"

    if url_cache_path.exists():
        return url_cache_path.read_text(encoding="utf-8").strip(), duration_sec

    if not trimmed_path.exists():
        _trim_audio_to_wav(source_path, trimmed_path, REF_AUDIO_MAX_SECONDS)

    public_url = _upload_to_storage(
        trimmed_path,
        REF_AUDIO_TRIM_STORAGE_FOLDER,
        storage_file_name=f"ref_trim_{url_hash}",
    )
    url_cache_path.write_text(public_url, encoding="utf-8")
    return public_url, duration_sec


def wav_bytes_to_audio(data: bytes) -> dict:
    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def generate_voice_clone(
    api_base_url: str,
    api_key: str,
    speech_text: str,
    ref_audio_url: str,
    ref_text: str,
    guidance_scale: float = 2.0,
    num_step: int = 32,
    timeout: float = 20.0,
) -> dict:
    ref_audio_for_api, ref_duration_sec = _prepare_ref_audio_url(ref_audio_url)
    ref_text_for_api = _trim_ref_text(ref_text, ref_duration_sec, REF_AUDIO_MAX_SECONDS)

    base_url = api_base_url.rstrip("/")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url}/audio/speech",
        json={
            "input": speech_text,
            "references": [
                {
                    "audio_path": ref_audio_for_api,
                    "text": ref_text_for_api,
                }
            ],
            "extra_params": {
                "guidance_scale": float(guidance_scale),
                "num_step": int(num_step),
            },
        },
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return wav_bytes_to_audio(resp.content)
