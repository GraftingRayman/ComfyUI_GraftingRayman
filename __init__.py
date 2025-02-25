PREFIX = '\33[31m*\33[94m Loading GraftingRaymans GR Nodes \33[31m*\33[0m '
print(f"\n\n\33[31m************************************\33[0m")
print(f"{PREFIX}\n\33[31m************************************")

from .Nodes.GRImage import GRImageSize, GRImageResize, GRStackImage, GRResizeImageMethods, GRImageDetailsDisplayer, GRImageDetailsSave, GRImagePaste, GRImagePasteWithMask, GRBackgroundRemoverREMBG
from .Nodes.GRPrompt import GRPromptSelector, GRPromptSelectorMulti, GRPromptHub, GRPrompty, GRPromptGen
from .Nodes.GRMask import GRMaskResize, GRMaskCreate, GRMultiMaskCreate, GRMaskCreateRandom, GRImageMask, GRMaskCreateRandomMulti, GRMask
from .Nodes.GRTile import GRTileImage, GRTileFlipImage, GRFlipTileRedRing, GRFlipTileInverted, GRCheckeredBoard
from .Nodes.GRTextOverlay import GRTextOverlay, GROnomatopoeia
from .Nodes.GRCounter import GRCounterVideo
from .Nodes.GRScroller import GRScrollerVideo
from .Nodes.GRPanOrZoom import GRPanOrZoom
from .Nodes.GRPromptGenExtended import GRPromptGenExtended
from .Nodes.GRBLIPL2CaptionGenerator import GRBLIP2CaptionGenerator

import time


NODE_CLASS_MAPPINGS = { "GR Prompt Selector" : GRPromptSelector , "GR Image Resize" : GRImageResize, "GR Mask Resize" : GRMaskResize, "GR Mask Create" : GRMaskCreate, "GR Multi Mask Create" : GRMultiMaskCreate, "GR Image Size": GRImageSize, "GR Tile and Border Image": GRTileImage, "GR Prompt Selector Multi": GRPromptSelectorMulti, "GR Tile and Border Image Random Flip" : GRTileFlipImage, "GR Mask Create Random": GRMaskCreateRandom, "GR Stack Image": GRStackImage, "GR Image Resize Methods" : GRResizeImageMethods,"GR Image Details Displayer": GRImageDetailsDisplayer, "GR Image Details Saver": GRImageDetailsSave, "GR Flip Tile Random Red Ring": GRFlipTileRedRing, "GR Flip Tile Random Inverted": GRFlipTileInverted,"GR Text Overlay": GRTextOverlay,"GR Image/Depth Mask": GRImageMask,"GR Checkered Board":GRCheckeredBoard, "GR Onomatopoeia": GROnomatopoeia, "GR Image Paste": GRImagePaste, "GR Prompt HUB": GRPromptHub, "GR Image Paste With Mask": GRImagePasteWithMask, "GR Counter": GRCounterVideo, "GR Background Remover REMBG": GRBackgroundRemoverREMBG, "GR Scroller": GRScrollerVideo, "GR Mask Create Random Multi": GRMaskCreateRandomMulti, "GR Mask": GRMask, "GR Prompty": GRPrompty, "GR Pan Or Zoom":GRPanOrZoom, "GR Prompt Generator": GRPromptGen, "GR Prompt Generator Extended": GRPromptGenExtended, "GR BLIP 2 Caption Generator": GRBLIP2CaptionGenerator }

NODE_DISPLAY_NAME_MAPPINGS = { }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

for i in range(10):
    time.sleep(0.1) 
    print ("\r Loading... {}".format(i)+str(i), end="")

print(f"\b\b\b\b\b\b\b\b\b\b\b\b\b\b*\33[0m     22 Grafting Nodes Loaded     \33[31m*")
print(f"*\33[0m           Get Grafting           \33[31m*")
print(f"\33[31m************************************\33[0m\n\n")
