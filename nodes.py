import os

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
        }

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
        prompts = positive
        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)
        return ([[condP, {"pooled_output": pooledP}]],[[condN, {"pooled_output": pooledP}]], positive )

