from typing import Any, Dict
import numpy as np
import random

class GRImageDimensions:
    PRESET_DIMENSIONS = {
        "(SD) 512x512": (512, 512),
        "640x480 (VGA)": (640, 480),
        "(SD2) 768x512": (768, 512),
        "(SD2) 768x768": (768, 768),
        "800x600 (SVGA)": (800, 600),
        "960x544 (Half HD)": (960, 544),
        "1024x768 (XGA)": (1024, 768),
        "(SDXL) 1024x1024": (1024, 1024),
        "1280x720 (HD)": (1280, 720),
        "1366x768 (HD)": (1366, 768),
        "1600x900 (HD+)": (1600, 900),
        "1920x1080 (Full HD)": (1920, 1080),
        "2560x1440 (Quad HD)": (2560, 1440),
        "3840x2160 (4K UHD)": (3840, 2160),
        "5120x2880 (5K)": (5120, 2880),
        "7680x4320 (8K)": (7680, 4320),
        "480x640 (VGA Portrait)": (480, 640),
        "480x832 (Portrait)": (480, 832),
        "(SD2) 512x768 (Portrait)": (512, 768),
        "544x960 (Half HD Portrait)": (544, 960),
        "600x800 (SVGA Portrait)": (600, 800),
        "720x1280 (HD Portrait)": (720, 1280),
        "768x1024 (XGA Portrait)": (768, 1024),
        "768x1366 (HD Portrait)": (768, 1366),
        "832x480 (Portrait)": (832, 480),
        "900x1600 (HD+ Portrait)": (900, 1600),
        "1080x1920 (Full HD Portrait)": (1080, 1920),
        "1440x2560 (Quad HD Portrait)": (1440, 2560),
        "2160x3840 (4K UHD Portrait)": (2160, 3840),
        "2880x5120 (5K Portrait)": (2880, 5120),
        "4320x7680 (8K Portrait)": (4320, 7680),
        # Square additions
        "720x720 (Square)": (720, 720),
        "1080x1080 (Square)": (1080, 1080),
        "1440x1440 (Square)": (1440, 1440),
        "2160x2160 (Square)": (2160, 2160),
        "2880x2880 (Square)": (2880, 2880),
        "4320x4320 (Square)": (4320, 4320),
        "5120x5120 (Square)": (5120, 5120),
        "7680x7680 (Square)": (7680, 7680),
    }

    @classmethod
    def get_preset_keys_with_none(cls):
        return ["NONE"] + sorted(cls.PRESET_DIMENSIONS.keys(),
                                 key=lambda k: cls.PRESET_DIMENSIONS[k][0] * cls.PRESET_DIMENSIONS[k][1])

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        keys_with_none = cls.get_preset_keys_with_none()
        return {
            "required": {
                "preset": (keys_with_none[1:],),  # exclude NONE
                "random": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0}),
                "start": (keys_with_none,),
                "end": (keys_with_none,),
            },
            "optional": {
                "image": ("IMAGE", {"default": None, "optional": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "utils"

    def get_dimensions(self, preset: str, random: bool, seed: int,
                       start: str, end: str, image: np.ndarray = None):
        # Case 1: Image provided overrides everything
        if image is not None:
            if len(image.shape) == 4:  # batch
                h, w, _ = image[0].shape  # use first image
            elif len(image.shape) == 3:  # single
                h, w, _ = image.shape
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            # Determine pixel area range for start/end
            items = list(self.PRESET_DIMENSIONS.items())
            start_area = 0 if start == "NONE" else self.PRESET_DIMENSIONS[start][0] * self.PRESET_DIMENSIONS[start][1]
            end_area = float('inf') if end == "NONE" else self.PRESET_DIMENSIONS[end][0] * self.PRESET_DIMENSIONS[end][1]

            filtered_presets = [(k, v) for k, v in items if start_area <= v[0] * v[1] <= end_area]
            if not filtered_presets:
                raise ValueError("No resolutions available in the selected start/end range.")

            if random:
                random.seed(seed)
                preset, (w, h) = random.choice(filtered_presets)
            else:
                w, h = self.PRESET_DIMENSIONS[preset]

        return (w, h)



