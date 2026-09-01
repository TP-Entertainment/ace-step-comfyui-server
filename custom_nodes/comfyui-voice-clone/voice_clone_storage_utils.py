from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import requests
import torch
import torchaudio

AUDIO_CACHE_DIR = Path("temp/voice_clone_audio")
AUDIO_UPLOAD_DIR = Path("temp/voice_clone_upload")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_project_imports() -> Path:
    root = _project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _download_file(file_url: str, save_path: str | Path, timeout: float = 60.0) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=2048 * 1024):
                if chunk:
                    f.write(chunk)

    return path


def _url_cache_path(audio_url: str) -> Path:
    url_hash = hashlib.md5(audio_url.encode("utf-8")).hexdigest()
    suffix = Path(audio_url.split("?", 1)[0]).suffix or ".audio"
    return AUDIO_CACHE_DIR / f"url_{url_hash}{suffix}"


def load_audio_from_url(audio_url: str, timeout: float = 60.0) -> dict:
    audio_url = audio_url.strip()
    if not audio_url:
        raise ValueError("audio_url is required")

    cache_path = _url_cache_path(audio_url)
    if not cache_path.exists():
        _download_file(audio_url, cache_path, timeout=timeout)

    waveform, sample_rate = torchaudio.load(str(cache_path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def upload_audio_to_storage(
    audio: dict,
    storage_folder: str,
    storage_file_name: str = "",
) -> str:
    ensure_project_imports()
    from src.services.cloud_storage.google_storage import upload

    storage_folder = storage_folder.strip()
    if not storage_folder:
        raise ValueError("storage_folder is required")

    AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_name = storage_file_name.strip() or str(uuid.uuid4())
    temp_path = AUDIO_UPLOAD_DIR / f"{file_name}.wav"

    waveform = audio["waveform"].squeeze(0).cpu()
    torchaudio.save(str(temp_path), waveform, int(audio["sample_rate"]))

    try:
        return upload(temp_path, storage_folder, file_name)
    finally:
        temp_path.unlink(missing_ok=True)
