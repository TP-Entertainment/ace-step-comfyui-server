import hashlib

from .voice_clone_utils import generate_voice_clone


class VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base_url": (
                    "STRING",
                    {"default": "https://example.com/v1", "multiline": False},
                ),
                "api_key": ("STRING", {"default": ""}),
                "speech_text": (
                    "STRING",
                    {"default": "Hello world", "multiline": True},
                ),
                "ref_audio_url": (
                    "STRING",
                    {"default": "https://example.com/ref.wav", "multiline": False},
                ),
                "ref_audio_text": (
                    "STRING",
                    {"default": "Reference transcript", "multiline": True},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "num_step": (
                    "INT",
                    {"default": 32, "min": 1, "max": 128, "step": 1},
                ),
                "timeout": (
                    "FLOAT",
                    {"default": 20.0, "min": 1.0, "max": 600.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "audio/voice_clone"
    DESCRIPTION = "Gọi API TTS voice clone và trả về audio."

    def clone(
        self,
        api_base_url: str,
        api_key: str,
        speech_text: str,
        ref_audio_url: str,
        ref_audio_text: str,
        guidance_scale: float,
        num_step: int,
        timeout: float,
    ):
        audio = generate_voice_clone(
            api_base_url=api_base_url,
            api_key=api_key,
            speech_text=speech_text,
            ref_audio_url=ref_audio_url,
            ref_text=ref_audio_text,
            guidance_scale=guidance_scale,
            num_step=num_step,
            timeout=timeout,
        )
        return (audio,)

    @classmethod
    def IS_CHANGED(
        cls,
        api_base_url: str,
        api_key: str,
        speech_text: str,
        ref_audio_url: str,
        ref_audio_text: str,
        guidance_scale: float,
        num_step: int,
        timeout: float,
    ):
        payload = "|".join(
            [
                api_base_url,
                api_key,
                speech_text,
                ref_audio_url,
                ref_audio_text,
                str(guidance_scale),
                str(num_step),
                str(timeout),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
