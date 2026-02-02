import folder_paths
from comfy import sd


class GRLoraLoader:
    """
    A node that allows adding multiple LoRAs with custom weights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "apply_to_clip": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    CATEGORY = "loaders"

    def apply_loras(self, model, apply_to_clip=True, clip=None, unique_id=None, extra_pnginfo=None):
        """
        Apply all LoRAs from the saved configuration file
        """
        # print(f"[GRLoraLoader] apply_loras called")
        # print(f"[GRLoraLoader] Node ID: {unique_id}")
        # print(f"[GRLoraLoader] apply_to_clip: {apply_to_clip}")
        # print(f"[GRLoraLoader] clip provided: {clip is not None}")
        
        if not unique_id:
            print("[GRLoraLoader] No unique_id provided, cannot load LoRAs")
            return (model, clip)
        
        # Load configuration from file
        import os
        import json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "lora_configs", f"GRLoraLoader_{unique_id}.json")
        
        # print(f"[GRLoraLoader] Looking for config file: {config_file}")
        
        if not os.path.exists(config_file):
            # print(f"[GRLoraLoader] Config file not found, no LoRAs to apply")
            return (model, clip)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            lora_widgets = config.get('lora_widgets', [])
            # print(f"[GRLoraLoader] Loaded config with {len(lora_widgets)} LoRAs")
            
            # Apply each lora
            for idx, lora_data in enumerate(lora_widgets):
                if not lora_data.get("on", True):
                    # print(f"[GRLoraLoader] Skipping disabled LoRA: {lora_data.get('lora')}")
                    continue
                    
                lora_name = lora_data.get("lora")
                if not lora_name or lora_name == "None":
                    continue
                    
                strength_model = lora_data.get("strength", 1.0)
                strength_clip = lora_data.get("strengthTwo", strength_model)
                
                # If apply_to_clip is False or clip is not provided, set CLIP strength to 0
                if not apply_to_clip or clip is None:
                    strength_clip = 0.0
                
                # print(f"[GRLoraLoader] Applying LoRA {idx+1}/{len(lora_widgets)}: {lora_name} (model: {strength_model}, clip: {strength_clip})")
                
                # Load and apply the LoRA
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path:
                    try:
                        # Load the LoRA file
                        import comfy.utils
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        
                        # Apply to model and clip (if clip is provided)
                        if clip is not None:
                            model_lora, clip_lora = sd.load_lora_for_models(
                                model, 
                                clip, 
                                lora, 
                                strength_model, 
                                strength_clip
                            )
                            model = model_lora
                            clip = clip_lora
                        else:
                            # Only apply to model if no clip provided
                            model_lora, _ = sd.load_lora_for_models(
                                model, 
                                None, 
                                lora, 
                                strength_model, 
                                0.0
                            )
                            model = model_lora
                        # print(f"[GRLoraLoader] ✓ Successfully applied: {lora_name}")
                    except Exception as e:
                        print(f"[GRLoraLoader] ✗ Error loading LoRA {lora_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[GRLoraLoader] ✗ LoRA path not found: {lora_name}")
            
            # print(f"[GRLoraLoader] Finished applying LoRAs")
            
        except Exception as e:
            # print(f"[GRLoraLoader] Error reading config file: {e}")
            import traceback
            traceback.print_exc()
        
        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "GRLoraLoader": GRLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoraLoader": "GRLoraLoader"
}