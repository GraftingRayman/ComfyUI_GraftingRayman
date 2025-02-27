import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import comfy.model_management as mm
from PIL import Image  # Import PIL to create a dummy image

class BLIP2TextExpander:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "", "multiline": True}),  # Input text to be expanded
                "max_new_tokens": ("INT", {"default": 50, "min": 1, "max": 100}),  # Maximum number of tokens to generate
                "num_beams": ("INT", {"default": 5, "min": 1, "max": 10}),  # Number of beams for beam search
                "do_sample": ("BOOLEAN", {"default": True}),  # Whether to use sampling
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),  # Sampling temperature
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),  # Top-p sampling
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),  # Seed for reproducibility
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("expanded_text",)
    FUNCTION = "expand_text"
    CATEGORY = "GraftingRayman/Text Processing"

    def load_model(self, device):
        """Load the BLIP-2 processor and model."""
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
        return processor, model

    def expand_text(self, text_input, max_new_tokens, num_beams, do_sample, temperature, top_p, seed):
        """Expand the input text using BLIP-2."""
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Load the processor and model
        processor, model = self.load_model(device)

        # Set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # Create a dummy white image (224x224 is a common size for vision models)
        dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))  # White image

        # Prepare the input text and dummy image
        inputs = processor(dummy_image, text=text_input, return_tensors="pt").to(device)

        # Extract pixel_values from inputs
        pixel_values = inputs["pixel_values"]

        # Generate the expanded text
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,  # Pass pixel_values explicitly
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode the generated text
        expanded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Offload the model if needed
        model.to(offload_device)
        mm.soft_empty_cache()

        print(f"Expanded text: {expanded_text}")
        return (expanded_text,)
