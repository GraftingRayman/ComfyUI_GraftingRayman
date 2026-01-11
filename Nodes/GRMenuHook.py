class GRMenuHook:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "utils"

    def noop(self):
        return ()

NODE_CLASS_MAPPINGS = {
    "GRMenuHook": GRMenuHook
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRMenuHook": "🧩 GR Menu Hook"
}