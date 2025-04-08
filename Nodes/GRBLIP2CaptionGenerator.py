import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import logging
import folder_paths
import comfy.utils

class GRBLIP2CaptionGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Accepts an image tensor in [B, H, W, C] format
                "max_length": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "num_beams": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"

    CATEGORY = "GraftingRayman/Image Processing"

    def generate_caption(self, image, max_length, num_beams, temperature, top_k):
        """Generate a caption for the given image using BLIP-2."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

        # Convert the input tensor to a PIL Image
        if isinstance(image, torch.Tensor):
            # Ensure the tensor is in the correct shape [B, H, W, C] and normalized to [0, 1]
            if image.dim() == 4:  # Batch dimension is present
                image = image[0]  # Take the first image in the batch
            image = image.mul(255).byte().cpu().numpy()  # Convert to numpy array and scale to [0, 255]
            image = Image.fromarray(image)  # Convert to PIL Image

        # Process the image and generate the caption
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
            )
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"Generated caption: {caption}")
        return (caption,)
