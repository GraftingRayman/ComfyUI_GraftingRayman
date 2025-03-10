import os
import sys
import comfy.sd
import comfy.utils
import folder_paths
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

class GRLora:
    def __init__(self):
        self.loaded_loras = {}  # Cache for loaded LoRAs

    @classmethod
    def INPUT_TYPES(s):
        loras_root = folder_paths.get_folder_paths("loras")[0]  # Get the first loras directory
        subfolders = [name for name in os.listdir(loras_root) if os.path.isdir(os.path.join(loras_root, name))]
        subfolders.insert(0, "None")  # Add "None" to use the root loras directory
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subfolder": (subfolders, {"default": "None"}),  # Subfolder input
                "num_loras": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),  # Number of LoRAs to load
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),  # Seed for random selection
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Strength for all LoRAs
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "lora_info",)  # Only return MODEL, CLIP, and lora_info
    FUNCTION = "load_lora"
    CATEGORY = "GraftingRayman/LoRA"  # Directly assign the category as a string

    def load_lora(self, model, clip, subfolder, num_loras, seed, strength):
        loras_root = folder_paths.get_folder_paths("loras")[0]  # Get the first loras directory
        if subfolder != "None":
            loras_root = os.path.join(loras_root, subfolder)  # Append the subfolder if specified
        lora_files = [f for f in os.listdir(loras_root) if f.endswith((".safetensors", ".ckpt"))]
        if not lora_files:
            raise ValueError(f"No LoRA files found in {loras_root}")
        random.seed(seed)
        selected_loras = random.sample(lora_files, min(num_loras, len(lora_files)))
        weights = [random.random() for _ in selected_loras]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # Normalize to sum to 1.0
        lora_info = "Selected LoRAs and Weights:\n"
        for lora_name, weight in zip(selected_loras, weights):
            lora_info += f"- {lora_name}: {weight:.2f}\n"
        model_lora, clip_lora = model, clip
        for lora_name, weight in zip(selected_loras, weights):
            lora_path = os.path.join(loras_root, lora_name)
            lora = None
            if lora_path in self.loaded_loras:
                lora = self.loaded_loras[lora_path]
            else:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_path] = lora
            strength_lora = strength * weight
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_lora, strength_lora)

        return (model_lora, clip_lora, lora_info)