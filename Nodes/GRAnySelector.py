class GRAnySelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ANY_1": ("*",),
                "ANY_2": ("*",),
                "ANY_3": ("*",),
                "ANY_4": ("*",),
                "ANY_5": ("*",),
                "ANY_6": ("*",),
                "ANY_7": ("*",),
                "ANY_8": ("*",),
                "ANY_9": ("*",),
                "ANY_10": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "select_any"
    CATEGORY = "utils"

    def select_any(self, **kwargs):
        for i in range(1, 11):
            value = kwargs.get(f"ANY_{i}", None)
            if value is not None:
                return (value,)

        return (None,)


NODE_CLASS_MAPPINGS = {
    "GRAnySelector": GRAnySelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRAnySelector": "GRAnySelector"
}
