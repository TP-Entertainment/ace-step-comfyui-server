import hashlib

from .audio_process_utils import postprocess_generated_audio, preprocess_reference_audio


class VoiceClonePreprocessRefAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_sample_rate": (
                    "INT",
                    {"default": 24000, "min": 8000, "max": 96000, "step": 1000},
                ),
                "max_duration": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 120.0, "step": 0.1},
                ),
                "remove_silence": ("BOOLEAN", {"default": True}),
                "mid_silence_ms": (
                    "INT",
                    {"default": 200, "min": 0, "max": 5000, "step": 10},
                ),
                "lead_silence_ms": (
                    "INT",
                    {"default": 100, "min": 0, "max": 5000, "step": 10},
                ),
                "trail_silence_ms": (
                    "INT",
                    {"default": 200, "min": 0, "max": 5000, "step": 10},
                ),
                "normalize_quiet": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "ref_rms")
    FUNCTION = "preprocess"
    CATEGORY = "audio/voice_clone"
    DESCRIPTION = (
        "Preprocess reference audio for voice clone: mono, resample, "
        "optional silence removal, quiet boost, then hard trim to max_duration."
    )

    def preprocess(
        self,
        audio,
        target_sample_rate,
        max_duration,
        remove_silence,
        mid_silence_ms,
        lead_silence_ms,
        trail_silence_ms,
        normalize_quiet,
    ):
        processed, ref_rms = preprocess_reference_audio(
            audio=audio,
            target_sample_rate=target_sample_rate,
            max_duration=max_duration,
            remove_silence_enabled=remove_silence,
            mid_sil_ms=mid_silence_ms,
            lead_sil_ms=lead_silence_ms,
            trail_sil_ms=trail_silence_ms,
            normalize_quiet=normalize_quiet,
        )
        return (processed, ref_rms)

    @classmethod
    def IS_CHANGED(
        cls,
        audio,
        target_sample_rate,
        max_duration,
        remove_silence,
        mid_silence_ms,
        lead_silence_ms,
        trail_silence_ms,
        normalize_quiet,
    ):
        payload = "|".join(
            [
                str(audio["sample_rate"]),
                str(audio["waveform"].shape),
                str(float(audio["waveform"].sum().item())),
                str(target_sample_rate),
                str(max_duration),
                str(remove_silence),
                str(mid_silence_ms),
                str(lead_silence_ms),
                str(trail_silence_ms),
                str(normalize_quiet),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VoiceClonePostprocessAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "ref_rms": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "RMS from preprocess node. 0 = peak normalize to 0.5.",
                    },
                ),
                "remove_silence": ("BOOLEAN", {"default": True}),
                "mid_silence_ms": (
                    "INT",
                    {"default": 500, "min": 0, "max": 5000, "step": 10},
                ),
                "lead_silence_ms": (
                    "INT",
                    {"default": 100, "min": 0, "max": 5000, "step": 10},
                ),
                "trail_silence_ms": (
                    "INT",
                    {"default": 100, "min": 0, "max": 5000, "step": 10},
                ),
                "fade_duration": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "pad_duration": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "postprocess"
    CATEGORY = "audio/voice_clone"
    DESCRIPTION = (
        "Postprocess generated audio: optional silence removal, volume match, "
        "fade in/out and edge padding (OmniVoice-style output cleanup)."
    )

    def postprocess(
        self,
        audio,
        ref_rms,
        remove_silence,
        mid_silence_ms,
        lead_silence_ms,
        trail_silence_ms,
        fade_duration,
        pad_duration,
    ):
        processed = postprocess_generated_audio(
            audio=audio,
            ref_rms=ref_rms,
            remove_silence_enabled=remove_silence,
            mid_sil_ms=mid_silence_ms,
            lead_sil_ms=lead_silence_ms,
            trail_sil_ms=trail_silence_ms,
            fade_duration=fade_duration,
            pad_duration=pad_duration,
        )
        return (processed,)

    @classmethod
    def IS_CHANGED(
        cls,
        audio,
        ref_rms,
        remove_silence,
        mid_silence_ms,
        lead_silence_ms,
        trail_silence_ms,
        fade_duration,
        pad_duration,
    ):
        payload = "|".join(
            [
                str(audio["sample_rate"]),
                str(audio["waveform"].shape),
                str(float(audio["waveform"].sum().item())),
                str(ref_rms),
                str(remove_silence),
                str(mid_silence_ms),
                str(lead_silence_ms),
                str(trail_silence_ms),
                str(fade_duration),
                str(pad_duration),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
