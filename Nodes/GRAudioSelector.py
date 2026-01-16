class GRAudioSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
                "audio_9": ("AUDIO",),
                "audio_10": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "select_audio"
    CATEGORY = "audio/utils"

    def select_audio(self, **kwargs):
        # Return the first non-None audio
        for i in range(1, 11):
            audio = kwargs.get(f"audio_{i}")
            if audio is not None:
                # audio is already a tensor, could be shape [1, channels, samples] or [batch, channels, samples]
                # Just pass it through as-is
                return (audio,)
        return (None,)

NODE_CLASS_MAPPINGS = {
    "GRAudioSelector": GRAudioSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRAudioSelector": "GRAudioSelector"
}