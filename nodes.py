import os
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import latent_preview
from clip import tokenize, model
from PIL import Image, ImageDraw

class GRPromptSelector:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "positive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a2": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "positive_a3": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a4": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a5": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "positive_a6": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "always_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "negative_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","STRING",)
    RETURN_NAMES = ("positive","negative","prompts",)
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman"
        


    def select_prompt(self, clip, positive_a1, positive_a2, positive_a3, positive_a4, positive_a5, positive_a6, always_a1, negative_a1, select_prompt):

        if select_prompt == 1:
            clipa = positive_a1
        elif select_prompt == 2:
            clipa = positive_a2
        elif select_prompt == 3:
            clipa = positive_a3
        elif select_prompt == 4:
            clipa = positive_a4
        elif select_prompt == 5:
            clipa = positive_a5
        elif select_prompt == 6:
            clipa = positive_a6
        positive = clipa + ", " + always_a1
        prompts = "positive:\n" + positive + "\n\nnegative: \n" + negative_a1
        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)
        return ([[condP, {"pooled_output": pooledP}]],[[condN, {"pooled_output": pooledP}]], prompts )



class GRImageResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "GraftingRayman"

    def resize_image(self, image, height, width):
        input_image = image.permute((0, 3, 1, 2))
        resized_image = TF.resize(input_image, (height, width))
        resized_image = resized_image.permute((0, 2, 3, 1))
        return (resized_image,)
        


class GRMaskResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "resize_mask"
    CATEGORY = "GraftingRayman"

    def resize_mask(self, mask, height, width):
        resized_mask = TF.resize(mask, (height, width))
        return resized_mask,



class GRMaskCreate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "mask_width": ("FLOAT", {"min": 0, "max": 1}),
                "position_percentage": ("FLOAT", {"min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "GraftingRayman"

    def create_mask(self, height, width, mask_width, position_percentage):
        transparent_width = int(width * mask_width)
        position = int(width * position_percentage)
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        x_start = max(0, min(width - transparent_width, position))
        mask[:, :, :, x_start:x_start + transparent_width] = 1.
        return mask


class GRMultiMaskCreate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "num_masks": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("MASK",) * 8  
    RETURN_NAMES = ("mask1", "mask2", "mask3", "mask4", "mask5", "mask6", "mask7", "mask8")
    FUNCTION = "create_masks"
    CATEGORY = "GraftingRayman"

    def create_masks(self, height, width, num_masks):
        masks = []
        transparent_width = width // num_masks
        for i in range(num_masks):
            mask = torch.zeros((1, height, width), dtype=torch.float32)
            start_x = i * transparent_width
            end_x = (i + 1) * transparent_width if i < num_masks - 1 else width
            mask[:, :, start_x:end_x] = 1.
            masks.append(mask)
        return tuple(masks)


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