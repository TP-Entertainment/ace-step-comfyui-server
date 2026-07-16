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
    "acestep_v1.5_xl_turbo_bf16": {
        "url": (
            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
            "resolve/main/split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors"
        ),
        "filename": "acestep_v1.5_xl_turbo_bf16.safetensors",
        "subdir": "models/diffusion_models",
        "checksum": "e879038e7b9cbdcf260caf1e01ee1b39625081056ed55b20a6530d57b245c79b",
    },
    "qwen_0.6b_ace15": {
        "url": (
            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
            "resolve/main/split_files/text_encoders/qwen_0.6b_ace15.safetensors"
        ),
        "filename": "qwen_0.6b_ace15.safetensors",
        "subdir": "modules/comfyui-server/models/text_encoders",
        "checksum": "bb32032673ff1cccbaebbbfc421b03c4ae3b8e3d1b46458560d783639616402f",
    },
    "qwen_4b_ace15": {
        "url": (
            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
            "resolve/main/split_files/text_encoders/qwen_4b_ace15.safetensors"
        ),
        "filename": "qwen_4b_ace15.safetensors",
        "subdir": "modules/comfyui-server/models/text_encoders",
        "checksum": "c98e2d7e3b70627e21d8a233a5f5927738d0a8f539b9bf7d3b83ac36e3e87a6f",
    },
    "ace_1.5_vae": {
        "url": (
            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
            "resolve/main/split_files/vae/ace_1.5_vae.safetensors"
        ),
        "filename": "ace_1.5_vae.safetensors",
        "subdir": "modules/comfyui-server/models/vae",
        "checksum": "7e2fdeb52ee2b212fb835d95ffb14160309045d0b975477a6cd97118b1370b08",
    },
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