class GRImageSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_image"
    CATEGORY = "image/utils"

    def select_image(self, **kwargs):
        # Return the first non-None image (can be single or batch)
        for i in range(1, 11):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                # img is already a tensor, could be shape [1, H, W, C] or [N, H, W, C]
                # Just pass it through as-is
                return (img,)
        return (None,)

NODE_CLASS_MAPPINGS = {
    "GRImageSelector": GRImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRImageSelector": "GRImageSelector"
}