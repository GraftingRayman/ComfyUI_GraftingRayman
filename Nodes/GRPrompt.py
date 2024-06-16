import os
from clip import tokenize


class GRPromptSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        clip_type = ("CLIP",)
        string_type = ("STRING", {"multiline": True, "dynamicPrompts": True})
        return {"required": {
            "clip": clip_type,
            **{f"positive_a{i}": string_type for i in range(1, 7)},
            "always_a1": string_type,
            "negative_a1": string_type,
            "select_prompt": ("INT", {"default": 1, "min": 1, "max": 6}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompts")
    FUNCTION = "select_prompt"
    CATEGORY = "GraftingRayman"

    def select_prompt(self, clip, **kwargs):
        select_prompt = kwargs["select_prompt"]
        positive_clip = kwargs[f"positive_a{select_prompt}"]
        always_a1 = kwargs["always_a1"]
        negative_a1 = kwargs["negative_a1"]

        positive = f"{positive_clip}, {always_a1}"
        prompts = f"positive:\n{positive}\n\nnegative:\n{negative_a1}"

        tokensP = clip.tokenize(positive)
        tokensN = clip.tokenize(negative_a1)
        condP, pooledP = clip.encode_from_tokens(tokensP, return_pooled=True)
        condN, pooledN = clip.encode_from_tokens(tokensN, return_pooled=True)

        return ([[condP, {"pooled_output": pooledP}]], [[condN, {"pooled_output": pooledN}]], prompts)

class GRPromptSelectorMulti:
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
            "alwayspositive_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            "negative_a1": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ), 
            }}


    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive1","positive2","positive3","positive4","positive5","positive6","negative",)
    FUNCTION = "select_promptmulti"
    CATEGORY = "GraftingRayman"
        


    def select_promptmulti(self, clip, positive_a1, positive_a2, positive_a3, positive_a4, positive_a5, positive_a6, alwayspositive_a1, negative_a1):

        positive1 = positive_a1 + ", " + alwayspositive_a1
        positive2 = positive_a2 + ", " + alwayspositive_a1
        positive3 = positive_a3 + ", " + alwayspositive_a1
        positive4 = positive_a4 + ", " + alwayspositive_a1
        positive5 = positive_a5 + ", " + alwayspositive_a1
        positive6 = positive_a6 + ", " + alwayspositive_a1
        tokensP1 = clip.tokenize(positive1)
        tokensP2 = clip.tokenize(positive2)
        tokensP3 = clip.tokenize(positive3)
        tokensP4 = clip.tokenize(positive4)
        tokensP5 = clip.tokenize(positive5)
        tokensP6 = clip.tokenize(positive6)
        tokensN1 = clip.tokenize(negative_a1)
        condP1, pooledP1 = clip.encode_from_tokens(tokensP1, return_pooled=True)
        condP2, pooledP2 = clip.encode_from_tokens(tokensP2, return_pooled=True)
        condP3, pooledP3 = clip.encode_from_tokens(tokensP3, return_pooled=True)
        condP4, pooledP4 = clip.encode_from_tokens(tokensP4, return_pooled=True)
        condP5, pooledP5 = clip.encode_from_tokens(tokensP5, return_pooled=True)
        condP6, pooledP6 = clip.encode_from_tokens(tokensP6, return_pooled=True)
        condN1, pooledN1 = clip.encode_from_tokens(tokensN1, return_pooled=True)

        return ([[condP1, {"pooled_output": pooledP1}]],[[condP2, {"pooled_output": pooledP2}]],[[condP3, {"pooled_output": pooledP3}]],[[condP4, {"pooled_output": pooledP4}]],[[condP5, {"pooled_output": pooledP5}]],[[condP6, {"pooled_output": pooledP6}]],[[condN1, {"pooled_output": pooledN1}]],)

