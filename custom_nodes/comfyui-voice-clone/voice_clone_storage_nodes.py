import hashlib

from .audio_process_utils import audio_fingerprint
from .voice_clone_storage_utils import load_audio_from_url, upload_audio_to_storage


class VoiceCloneLoadAudioFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_url": (
                    "STRING",
                    {"default": "https://example.com/audio.wav", "multiline": False},
                ),
                "timeout": (
                    "FLOAT",
                    {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_url")
    FUNCTION = "load"
    CATEGORY = "audio/voice_clone"
    DESCRIPTION = "Download audio from URL and return ComfyUI AUDIO."

    def load(self, audio_url: str, timeout: float):
        audio = load_audio_from_url(audio_url, timeout=timeout)
        return (audio, audio_url.strip())

    @classmethod
    def IS_CHANGED(cls, audio_url: str, timeout: float):
        payload = f"{audio_url.strip()}|{timeout}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VoiceCloneUploadAudioToStorage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "storage_folder": (
                    "STRING",
                    {"default": "voice_ref_trim", "multiline": False},
                ),
                "storage_file_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Leave empty to auto-generate UUID filename.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_url",)
    FUNCTION = "upload"
    CATEGORY = "audio/voice_clone"
    DESCRIPTION = (
        "Upload ComfyUI AUDIO to cloud storage via src.services.cloud_storage."
    )

    def upload(self, audio, storage_folder: str, storage_file_name: str):
        url = upload_audio_to_storage(
            audio=audio,
            storage_folder=storage_folder,
            storage_file_name=storage_file_name,
        )
        return (url,)

    @classmethod
    def IS_CHANGED(cls, audio, storage_folder: str, storage_file_name: str):
        payload = "|".join(
            [
                audio_fingerprint(audio),
                storage_folder.strip(),
                storage_file_name.strip(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
