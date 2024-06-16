from .GRnodes import GRImageResize, GRMaskResize, GRMaskCreate, GRMultiMaskCreate,  GRTileImage, GRTileFlipImage,  GRStackImage, GRResizeImageMethods, GRImageDetailsDisplayer, GRImageDetailsSave, GRFlipTileRedRing, GRFlipTileInverted
from .GRMaskCreateRandom import GRMaskCreateRandom
from .GRImageSize import GRImageSize
from .GRPromptSelector import GRPromptSelector, GRPromptSelectorMulti


NODE_CLASS_MAPPINGS = { "GR Prompt Selector" : GRPromptSelector , "GR Image Resize" : GRImageResize, "GR Mask Resize" : GRMaskResize, "GR Mask Create" : GRMaskCreate, "GR Multi Mask Create" : GRMultiMaskCreate, "GR Image Size": GRImageSize, "GR Tile and Border Image": GRTileImage, "GR Prompt Selector Multi": GRPromptSelectorMulti, "GR Tile and Border Image Random Flip" : GRTileFlipImage, "GR Mask Create Random": GRMaskCreateRandom, "GR Stack Image": GRStackImage, "GR Image Resize Methods" : GRResizeImageMethods,"GR Image Details Displayer": GRImageDetailsDisplayer, "GR Image Details Saver": GRImageDetailsSave, "GR Flip Tile Random Red Ring": GRFlipTileRedRing, "GR Flip Tile Random Inverted": GRFlipTileInverted }

NODE_DISPLAY_NAME_MAPPINGS = { }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

