from .voice_clone_audio_nodes import VoiceClonePostprocessAudio, VoiceClonePreprocessRefAudio
from .voice_clone_node import VoiceClone

NODE_CLASS_MAPPINGS = {
    "VoiceClone": VoiceClone,
    "VoiceClonePreprocessRefAudio": VoiceClonePreprocessRefAudio,
    "VoiceClonePostprocessAudio": VoiceClonePostprocessAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceClone": "Voice Clone",
    "VoiceClonePreprocessRefAudio": "Voice Clone Preprocess Ref Audio",
    "VoiceClonePostprocessAudio": "Voice Clone Postprocess Audio",
}
