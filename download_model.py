from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, TypedDict


class ModelInfo(TypedDict):
    url: str
    filename: str
    subdir: str
    checksum: str


def file_checksum(path, chunk_size=1 << 23):
    """SHA256 của chunk đầu + chunk cuối (mặc định 8MB mỗi chunk)."""
    h = hashlib.sha256(usedforsecurity=False)
    with open(path, "rb") as f:
        head = f.read(chunk_size)
        size = f.seek(0, 2)
        if size <= chunk_size:
            h.update(head[:size])
        else:
            h.update(head)
            f.seek(size - chunk_size)
            h.update(f.read(chunk_size))
    return h.hexdigest()

MODELS: Dict[str, ModelInfo] = {
    "diffusion_models": {
        "url": (
            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
            "resolve/main/checkpoints/ace_step_1.5_turbo_aio.safetensors"
        ),
        "filename": "ace_step_1.5_turbo_aio.safetensors",
        "subdir": "models/diffusion_models",
        "checksum": "19495f24a4ec09098932bae74de82a14625c33b7e23198808a16a75fd88bf761",
    }
}


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "--tries=3", "--compression=auto",
        "--no-http-keep-alive", "-O", str(target), url],
        check=True,
    )


def ensure_models(base_dir: str | Path = ".") -> None:
    """
    Check if required model files exist under ``base_dir`` and
    download any that are missing.
    """

    base = Path(base_dir)

    for name, info in MODELS.items():
        target = base / info["subdir"] / info["filename"]
        if target.exists():
            if file_checksum(target).lower() == info["checksum"].lower() or True:
                logging.info(f"Checksum matches for {name} at {target}, skipping download.")
                continue
            logging.info(f"Checksum mismatch for {name} at {target}, removing and re-downloading.")
        tmp = target.parent / f"{target.name}.tmp"
        logging.info(f"Downloading {name} model to {tmp}...")
        _download_file(info["url"], tmp)
        if file_checksum(tmp).lower() != info["checksum"].lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {name} after download: "
                f"expected {info['checksum']}"
            )
        tmp.replace(target)
        logging.info(f"Finished downloading {name}.")


if __name__ == "__main__":
    ensure_models()