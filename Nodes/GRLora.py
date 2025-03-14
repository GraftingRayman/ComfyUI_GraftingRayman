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
        loras_root = folder_paths.get_folder_paths("loras")[0]  # Get the first loras directory (parent folder)
        subfolders = [name for name in os.listdir(loras_root) if os.path.isdir(os.path.join(loras_root, name))]
        subfolders.insert(0, "None")  # Add "None" to use the root loras directory
        
        # Recursively find all LoRA files in the parent folder and include their folder names
        lora_files = []
        for root, _, files in os.walk(loras_root):
            for file in files:
                if file.endswith((".safetensors", ".ckpt")):
                    # Get the relative path from the loras_root
                    rel_path = os.path.relpath(os.path.join(root, file), loras_root)
                    lora_files.append(rel_path)
        lora_files.insert(0, "None")  # Add "None" as the first option
        
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subfolder": (subfolders, {"default": "None"}),  # Subfolder input for random LoRAs
                "num_loras": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),  # Number of LoRAs to load
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Strength for random LoRAs
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),  # Seed for random selection
                "enable_style_lora": ("BOOLEAN", {"default": True}),  # Boolean to enable/disable all style LoRAs
                "style_lora_1": (lora_files, {"default": "None"}),  # Dropdown to select the first style LoRA
                "style_lora_1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight for the first style LoRA
                "style_lora_2": (lora_files, {"default": "None"}),  # Dropdown to select the second style LoRA
                "style_lora_2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight for the second style LoRA
                "style_lora_3": (lora_files, {"default": "None"}),  # Dropdown to select the third style LoRA
                "style_lora_3_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight for the third style LoRA
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "lora_info",)
    FUNCTION = "load_lora"
    CATEGORY = "GraftingRayman/LoRA"

    def load_lora(self, model, clip, subfolder, num_loras, strength, seed, enable_style_lora, 
                  style_lora_1, style_lora_1_strength, style_lora_2, style_lora_2_strength, 
                  style_lora_3, style_lora_3_strength):
        loras_root = folder_paths.get_folder_paths("loras")[0]  # Get the first loras directory (parent folder)
        lora_info = ""
        model_lora, clip_lora = model, clip

        # Handle random LoRA selection (search in the specified subfolder)
        random_loras_root = loras_root
        if subfolder != "None":
            random_loras_root = os.path.join(loras_root, subfolder)  # Append the subfolder if specified
        
        lora_files = [f for f in os.listdir(random_loras_root) if f.endswith((".safetensors", ".ckpt"))]
        if lora_files and num_loras > 0:
            random.seed(seed)
            selected_loras = random.sample(lora_files, min(num_loras, len(lora_files)))
            weights = [random.random() for _ in selected_loras]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]  # Normalize to sum to 1.0

            # Add random LoRAs to the info string
            lora_info += "Random Loras:\n"
            for lora_name, weight in zip(selected_loras, weights):
                lora_path = os.path.join(random_loras_root, lora_name)
                if lora_path in self.loaded_loras:
                    lora = self.loaded_loras[lora_path]
                else:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loras[lora_path] = lora
                strength_lora = strength * weight
                model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_lora, strength_lora)
                lora_info += f"{lora_name} {strength_lora:.2f}\n"

        # Handle style LoRAs (if enabled)
        if enable_style_lora:
            style_loras = [
                (style_lora_1, style_lora_1_strength),
                (style_lora_2, style_lora_2_strength),
                (style_lora_3, style_lora_3_strength),
            ]
            # Add style LoRAs to the info string
            if lora_info:  # Add an empty line if random LoRAs were added
                lora_info += "\n"
            lora_info += "Style Loras:\n"
            for lora_name, lora_weight in style_loras:
                if lora_name != "None":
                    lora_path = os.path.join(loras_root, lora_name)
                    if lora_path in self.loaded_loras:
                        lora = self.loaded_loras[lora_path]
                    else:
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_path] = lora
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, lora_weight, lora_weight)
                    lora_info += f"{lora_name} {lora_weight:.2f}\n"

        if not lora_info:
            lora_info = "No LoRAs applied."

        return (model_lora, clip_lora, lora_info)