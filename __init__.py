PREFIX = '\33[31m*\33[94m Loading GraftingRaymans GR Nodes \33[31m*\33[0m '
print(f"\n\n\33[31m************************************\33[0m")
print(f"{PREFIX}\n\33[31m************************************")

from .Nodes.GRImage import GRImageSize, GRImageResize, GRStackImage, GRResizeImageMethods, GRImageDetailsDisplayer, GRImageDetailsSave
from .Nodes.GRPrompt import GRPromptSelector, GRPromptSelectorMulti
from .Nodes.GRMask import GRMaskResize, GRMaskCreate, GRMultiMaskCreate, GRMaskCreateRandom, GRImageMask
from .Nodes.GRTile import GRTileImage, GRTileFlipImage, GRFlipTileRedRing, GRFlipTileInverted
from .Nodes.GRTextOverlay import GRTextOverlay

import time


NODE_CLASS_MAPPINGS = { "GR Prompt Selector" : GRPromptSelector , "GR Image Resize" : GRImageResize, "GR Mask Resize" : GRMaskResize, "GR Mask Create" : GRMaskCreate, "GR Multi Mask Create" : GRMultiMaskCreate, "GR Image Size": GRImageSize, "GR Tile and Border Image": GRTileImage, "GR Prompt Selector Multi": GRPromptSelectorMulti, "GR Tile and Border Image Random Flip" : GRTileFlipImage, "GR Mask Create Random": GRMaskCreateRandom, "GR Stack Image": GRStackImage, "GR Image Resize Methods" : GRResizeImageMethods,"GR Image Details Displayer": GRImageDetailsDisplayer, "GR Image Details Saver": GRImageDetailsSave, "GR Flip Tile Random Red Ring": GRFlipTileRedRing, "GR Flip Tile Random Inverted": GRFlipTileInverted,"GR Text Overlay": GRTextOverlay,"GR Image/Depth Mask": GRImageMask }

NODE_DISPLAY_NAME_MAPPINGS = { }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

for i in range(10):
    time.sleep(0.1) 
    print ("\r Loading... {}".format(i)+str(i), end="")

print(f"\b\b\b\b\b\b\b\b\b\b\b\b\b\b*\33[0m     16 Grafting Nodes Loaded     \33[31m*")
print(f"*\33[0m           Get Grafting           \33[31m*")
print(f"\33[31m************************************\33[0m\n\n")