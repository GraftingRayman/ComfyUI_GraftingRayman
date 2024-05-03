import os

class GRPromptSelector:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "text_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "text_a2": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "text_a3": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "text_a4": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "text_a5": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "text_a6": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRaynman"
        


    def select_prompt(self, clip, text_a1, text_a2, text_a3, text_a4, text_a5, text_a6, select_prompt):

        if select_prompt == 1:
            clipa = text_a1
        elif select_prompt == 2:
            clipa = text_a2
        elif select_prompt == 3:
            clipa = text_a3
        elif select_prompt == 4:
            clipa = text_a4
        elif select_prompt == 5:
            clipa = text_a5
        elif select_prompt == 6:
            clipa = text_a6
        tokens = clip.tokenize(clipa)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled},]],)

