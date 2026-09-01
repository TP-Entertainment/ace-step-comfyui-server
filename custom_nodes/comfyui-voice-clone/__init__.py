from .voice_clone_audio_nodes import VoiceClonePostprocessAudio, VoiceClonePreprocessRefAudio
from .voice_clone_node import VoiceClone
from .voice_clone_storage_nodes import VoiceCloneLoadAudioFromUrl, VoiceCloneUploadAudioToStorage

NODE_CLASS_MAPPINGS = {
    "VoiceClone": VoiceClone,
    "VoiceClonePreprocessRefAudio": VoiceClonePreprocessRefAudio,
    "VoiceClonePostprocessAudio": VoiceClonePostprocessAudio,
    "VoiceCloneLoadAudioFromUrl": VoiceCloneLoadAudioFromUrl,
    "VoiceCloneUploadAudioToStorage": VoiceCloneUploadAudioToStorage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceClone": "Voice Clone",
    "VoiceClonePreprocessRefAudio": "Voice Clone Preprocess Ref Audio",
    "VoiceClonePostprocessAudio": "Voice Clone Postprocess Audio",
    "VoiceCloneLoadAudioFromUrl": "Voice Clone Load Audio From URL",
    "VoiceCloneUploadAudioToStorage": "Voice Clone Upload Audio To Storage",
}
