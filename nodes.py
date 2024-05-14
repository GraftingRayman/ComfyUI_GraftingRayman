import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image

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
                "height": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "ImageProcessing"

    def resize_image(self, image, height, width):
        input_image = image.permute((0, 3, 1, 2))
        resized_image = TF.resize(input_image, (height, width))
        resized_image = resized_image.permute((0, 2, 3, 1))
        return (resized_image,)



