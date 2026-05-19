import os
import server

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")


class GRLiveGroupController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, unique_id=None, extra_pnginfo=None):
        return {}


NODE_CLASS_MAPPINGS = {
    "GRLiveGroupController": GRLiveGroupController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLiveGroupController": "GR Live Group Controller",
}
