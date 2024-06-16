import torch

class GRImageSize:
    def __init__(self):
        pass
                
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "height": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
            "width": ("INT", {"default": 512, "min": 16, "max": 16000, "step": 8}),
            "standard": (["custom", "(SD) 512x512","(SDXL) 1024x1024","640x480 (VGA)", "800x600 (SVGA)", "960x544 (Half HD)","1024x768 (XGA)", "1280x720 (HD)", "1366x768 (HD)","1600x900 (HD+)","1920x1080 (Full HD or 1080p)","2560x1440 (Quad HD or 1440p)","3840x2160 (Ultra HD, 4K, or 2160p)","5120x2880 (5K)","7680x4320 (8K)"],),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },}

    RETURN_TYPES = ("INT","INT","LATENT")
    RETURN_NAMES = ("height","width","samples")
    FUNCTION = "image_size"
    CATEGORY = "GraftingRayman"
        


    def image_size(self, height, width, standard, batch_size=1):
        if standard == "custom":
            height = height
            width = width
        elif standard == "(SD) 512x512":
            width = 512
            height = 512
        elif standard == "(SDXL) 1024x1024":
            width = 1024
            height = 1024
        elif standard == "640x480 (VGA)":
            width = 640
            height = 480
        elif standard == "800x600 (SVGA)":
            width = 800
            height = 608
        elif standard == "960x544 (Half HD)":
            width = 960
            height = 544
        elif standard == "1024x768 (XGA)":
            width = 1024
            height = 768
        elif standard == "1280x720 (HD)":
            width = 1280
            height = 720
        elif standard == "1366x768 (HD)":
            width = 1360
            height = 768
        elif standard == "1600x900 (HD+)":
            width = 1600
            height = 896
        elif standard == "1920x1080 (Full HD or 1080p)":
            width = 1920
            height = 1088
        elif standard == "2560x1440 (Quad HD or 1440p)":
            width = 2560
            height = 1440
        elif standard == "3840x2160 (Ultra HD, 4K, or 2160p)":
            width = 3840
            height = 2160
        elif standard == "5120x2880 (5K)":
            width = 5120
            height = 2880
        elif standard == "7680x4320 (8K)":
            width = 7680
            height = 4320            
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
    
        return (height,width,{"samples":latent},)
