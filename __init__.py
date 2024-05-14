from .nodes import GRPromptSelector, GRImageResize, GRMaskResize
NODE_CLASS_MAPPINGS = { "GR Prompt Selector" : GRPromptSelector , "GR Image Resize" : GRImageResize, "GR Mask Resize" : GRMaskResize }
NODE_DISPLAY_NAME_MAPPINGS = { }
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

