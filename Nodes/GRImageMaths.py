class GRImageMultiplication:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiply": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "utils/image"

    def get_dimensions(self, image, multiply):
        # image shape: [batch, height, width, channels]
        _, height, width, _ = image.shape

        new_width = int(round(width * multiply))
        new_height = int(round(height * multiply))

        return (new_width, new_height)