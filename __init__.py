PREFIX = '\33[31m*\33[94m Loading GraftingRaymans GR Nodes \33[31m*\33[0m '
print(f"\n\n\33[31m************************************\33[0m")
print(f"{PREFIX}\n\33[31m************************************")

from .Nodes.GRImage import GRImageSize, GRImageResize, GRStackImage, GRResizeImageMethods, GRImageDetailsDisplayer, GRImageDetailsSave, GRImagePaste, GRImagePasteWithMask
from .Nodes.GRPrompt import GRPromptSelector, GRPromptSelectorMulti, GRPromptHub
from .Nodes.GRMask import GRMaskResize, GRMaskCreate, GRMultiMaskCreate, GRMaskCreateRandom, GRImageMask
from .Nodes.GRTile import GRTileImage, GRTileFlipImage, GRFlipTileRedRing, GRFlipTileInverted, GRCheckeredBoard
from .Nodes.GRTextOverlay import GRTextOverlay, GROnomatopoeia
from .Nodes.GRVideo import GRCounterVideo

import time


NODE_CLASS_MAPPINGS = { "GR Prompt Selector" : GRPromptSelector , "GR Image Resize" : GRImageResize, "GR Mask Resize" : GRMaskResize, "GR Mask Create" : GRMaskCreate, "GR Multi Mask Create" : GRMultiMaskCreate, "GR Image Size": GRImageSize, "GR Tile and Border Image": GRTileImage, "GR Prompt Selector Multi": GRPromptSelectorMulti, "GR Tile and Border Image Random Flip" : GRTileFlipImage, "GR Mask Create Random": GRMaskCreateRandom, "GR Stack Image": GRStackImage, "GR Image Resize Methods" : GRResizeImageMethods,"GR Image Details Displayer": GRImageDetailsDisplayer, "GR Image Details Saver": GRImageDetailsSave, "GR Flip Tile Random Red Ring": GRFlipTileRedRing, "GR Flip Tile Random Inverted": GRFlipTileInverted,"GR Text Overlay": GRTextOverlay,"GR Image/Depth Mask": GRImageMask,"GR Checkered Board":GRCheckeredBoard, "GR Onomatopoeia": GROnomatopoeia, "GR Image Paste": GRImagePaste, "GR Prompt HUB": GRPromptHub, "GR Image Paste With Mask": GRImagePasteWithMask, "GR Counter": GRCounterVideo }

NODE_DISPLAY_NAME_MAPPINGS = { }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

for i in range(10):
    time.sleep(0.1) 
    print ("\r Loading... {}".format(i)+str(i), end="")

print(f"\b\b\b\b\b\b\b\b\b\b\b\b\b\b*\33[0m     19 Grafting Nodes Loaded     \33[31m*")
print(f"*\33[0m           Get Grafting           \33[31m*")
print(f"\33[31m************************************\33[0m\n\n")
