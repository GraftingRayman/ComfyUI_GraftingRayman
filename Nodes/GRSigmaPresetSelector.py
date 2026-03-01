import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from enum import Enum
from PIL import Image
import json
import os

class PresetCategory(Enum):
    ZIMAGE = "zimage"
    FLUX = "flux"
    FLUX_KLEIN = "flux_klein"
    QWEN = "qwen"
    SDXL = "sdxl"
    PONY = "pony"
    SD = "sd"
    # Presets from original GRSigmas node
    BALANCED = "balanced"
    COMPOSITION_HEAVY = "composition_heavy"
    DETAIL_HEAVY = "detail_heavy"
    AGGRESSIVE = "aggressive"
    SUBTLE = "subtle"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    ABSTRACT = "abstract"
    FINE_DETAIL = "fine_detail"
    FAST_DECAY = "fast_decay"
    SLOW_DECAY = "slow_decay"
    MID_CENTRIC = "mid_centric"
    HIGH_CONTRAST = "high_contrast"
    LOW_CONTRAST = "low_contrast"
    # Manual presets from GRSigmaPresets
    ULTRA_LOCK = "ultra_lock"
    MICRO_DETAIL = "micro_detail"
    LOW_MOTION = "low_motion"
    IMG2IMG_SAFE = "img2img_safe"
    BALANCED_I2V = "balanced_i2v"
    STYLISED_MOTION = "stylised_motion"
    MANUAL_PRESET = "manual_preset"
    MID_SIGMA_FOCUS = "mid_sigma_focus"
    HIGH_DETAIL_TAIL = "high_detail_tail"
    EXPERIMENTAL_WIDE = "experimental_wide"

class GRSigmaPresetSelector:
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = [p.value for p in PresetCategory]
        
        return {
            "required": {
                "preset": (preset_list, {"default": PresetCategory.ZIMAGE.value}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "auto_scale_to_steps": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES = ("sigmas", "graph_preview", "preset_info")
    FUNCTION = "get_sigmas"
    CATEGORY = "GraftingRayman/Sigmas"

    def __init__(self):
        self.preset_sigmas = self._initialize_presets()
        
    def _initialize_presets(self):
        """Initialize all preset sigma schedules"""
        return {
            # ZIMAGE presets - optimized for Z-Image Turbo model
            PresetCategory.ZIMAGE.value: {
                "base": [0.991, 0.98, 0.92, 0.935, 0.90, 0.875, 0.750, 0.6582, 0.4556, 0.2000, 0.0000],
                "description": "Z-Image Turbo optimized schedule with 3-stage denoising",
                "zones": "Comp(3) Mid(5) Detail(3)",
                "range": "0.991 → 0.000"
            },
            
            # FLUX presets
            PresetCategory.FLUX.value: {
                "base": [1.00, 0.95, 0.85, 0.72, 0.58, 0.45, 0.32, 0.20, 0.10, 0.05, 0.00],
                "description": "Flux model optimized schedule with smooth decay",
                "zones": "Comp(4) Mid(4) Detail(3)",
                "range": "1.000 → 0.000"
            },
            
            # FLUX KLEIN presets
            PresetCategory.FLUX_KLEIN.value: {
                "base": [0.99, 0.92, 0.78, 0.65, 0.50, 0.38, 0.27, 0.18, 0.10, 0.04, 0.00],
                "description": "Flux Klein with faster initial decay for fine detail",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.990 → 0.000"
            },
            
            # QWEN presets
            PresetCategory.QWEN.value: {
                "base": [0.98, 0.94, 0.86, 0.75, 0.62, 0.48, 0.35, 0.24, 0.15, 0.07, 0.00],
                "description": "QWEN model with balanced structure-detail tradeoff",
                "zones": "Comp(4) Mid(4) Detail(3)",
                "range": "0.980 → 0.000"
            },
            
            # SDXL presets
            PresetCategory.SDXL.value: {
                "base": [0.95, 0.88, 0.78, 0.65, 0.52, 0.40, 0.30, 0.22, 0.15, 0.08, 0.00],
                "description": "SDXL standard denoising schedule",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.950 → 0.000"
            },
            
            # PONY presets
            PresetCategory.PONY.value: {
                "base": [0.96, 0.90, 0.80, 0.68, 0.55, 0.42, 0.31, 0.22, 0.14, 0.06, 0.00],
                "description": "Pony model with gentle transitions",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.960 → 0.000"
            },
            
            # SD presets (Stable Diffusion 1.5/2.1)
            PresetCategory.SD.value: {
                "base": [0.94, 0.86, 0.74, 0.60, 0.48, 0.36, 0.26, 0.18, 0.11, 0.05, 0.00],
                "description": "Standard SD 1.5/2.1 denoising schedule",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.940 → 0.000"
            },
            
            # Presets from original GRSigmas node
            PresetCategory.BALANCED.value: {
                "base": [1.0, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.00],
                "description": "Balanced composition and detail",
                "zones": "Comp(4) Mid(4) Detail(3)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.COMPOSITION_HEAVY.value: {
                "base": [1.0, 0.92, 0.85, 0.78, 0.70, 0.55, 0.40, 0.25, 0.12, 0.04, 0.00],
                "description": "Heavy focus on composition, less detail",
                "zones": "Comp(5) Mid(3) Detail(3)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.DETAIL_HEAVY.value: {
                "base": [0.98, 0.85, 0.72, 0.60, 0.48, 0.38, 0.30, 0.23, 0.16, 0.08, 0.00],
                "description": "Heavy focus on fine detail",
                "zones": "Comp(2) Mid(4) Detail(5)",
                "range": "0.980 → 0.000"
            },
            
            PresetCategory.AGGRESSIVE.value: {
                "base": [1.0, 0.88, 0.76, 0.64, 0.52, 0.40, 0.30, 0.20, 0.12, 0.05, 0.00],
                "description": "Aggressive denoising for strong changes",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.SUBTLE.value: {
                "base": [0.95, 0.86, 0.78, 0.70, 0.62, 0.54, 0.44, 0.34, 0.22, 0.10, 0.00],
                "description": "Subtle changes with smooth transitions",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.950 → 0.000"
            },
            
            PresetCategory.PORTRAIT.value: {
                "base": [0.96, 0.88, 0.80, 0.72, 0.62, 0.50, 0.38, 0.26, 0.16, 0.07, 0.00],
                "description": "Optimized for portrait generation",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.960 → 0.000"
            },
            
            PresetCategory.LANDSCAPE.value: {
                "base": [0.98, 0.92, 0.84, 0.74, 0.62, 0.48, 0.36, 0.26, 0.18, 0.09, 0.00],
                "description": "Optimized for landscape scenes",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.980 → 0.000"
            },
            
            PresetCategory.ARCHITECTURE.value: {
                "base": [0.99, 0.93, 0.85, 0.75, 0.63, 0.51, 0.39, 0.28, 0.18, 0.08, 0.00],
                "description": "Optimized for architectural images",
                "zones": "Comp(4) Mid(4) Detail(3)",
                "range": "0.990 → 0.000"
            },
            
            PresetCategory.ABSTRACT.value: {
                "base": [0.97, 0.89, 0.81, 0.73, 0.65, 0.55, 0.43, 0.31, 0.19, 0.08, 0.00],
                "description": "Creative abstract style",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.970 → 0.000"
            },
            
            PresetCategory.FINE_DETAIL.value: {
                "base": [0.93, 0.84, 0.73, 0.62, 0.51, 0.40, 0.30, 0.22, 0.15, 0.07, 0.00],
                "description": "Maximum fine detail preservation",
                "zones": "Comp(2) Mid(4) Detail(5)",
                "range": "0.930 → 0.000"
            },
            
            PresetCategory.FAST_DECAY.value: {
                "base": [1.0, 0.90, 0.78, 0.64, 0.50, 0.36, 0.24, 0.14, 0.07, 0.02, 0.00],
                "description": "Fast initial decay, long detail tail",
                "zones": "Comp(2) Mid(4) Detail(5)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.SLOW_DECAY.value: {
                "base": [0.96, 0.91, 0.86, 0.80, 0.72, 0.62, 0.50, 0.38, 0.24, 0.10, 0.00],
                "description": "Slow gradual decay throughout",
                "zones": "Comp(4) Mid(4) Detail(3)",
                "range": "0.960 → 0.000"
            },
            
            PresetCategory.MID_CENTRIC.value: {
                "base": [0.98, 0.90, 0.80, 0.68, 0.56, 0.44, 0.34, 0.24, 0.16, 0.08, 0.00],
                "description": "Focus on middle denoising stages",
                "zones": "Comp(3) Mid(5) Detail(3)",
                "range": "0.980 → 0.000"
            },
            
            PresetCategory.HIGH_CONTRAST.value: {
                "base": [1.0, 0.92, 0.82, 0.70, 0.58, 0.46, 0.34, 0.22, 0.12, 0.04, 0.00],
                "description": "High contrast transitions",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.LOW_CONTRAST.value: {
                "base": [0.94, 0.88, 0.82, 0.76, 0.68, 0.58, 0.46, 0.34, 0.22, 0.10, 0.00],
                "description": "Low contrast smooth transitions",
                "zones": "Comp(3) Mid(4) Detail(4)",
                "range": "0.940 → 0.000"
            },
            
            # Manual presets from GRSigmaPresets
            PresetCategory.ULTRA_LOCK.value: {
                "base": [0.0],
                "description": "Ultra Lock - No change, structure frozen",
                "zones": "Comp(1) Mid(0) Detail(0)",
                "range": "0.000 → 0.000"
            },
            
            PresetCategory.MICRO_DETAIL.value: {
                "base": [0.1, 0.0],
                "description": "Micro Detail - Texture polish only",
                "zones": "Comp(1) Mid(0) Detail(1)",
                "range": "0.100 → 0.000"
            },
            
            PresetCategory.LOW_MOTION.value: {
                "base": [0.2, 0.0],
                "description": "Low Motion - Subtle motion, strong lock",
                "zones": "Comp(1) Mid(0) Detail(1)",
                "range": "0.200 → 0.000"
            },
            
            PresetCategory.IMG2IMG_SAFE.value: {
                "base": [0.3, 0.0],
                "description": "Img2Img Safe - Classic stable img2img",
                "zones": "Comp(1) Mid(0) Detail(1)",
                "range": "0.300 → 0.000"
            },
            
            PresetCategory.BALANCED_I2V.value: {
                "base": [0.5, 0.25, 0.0],
                "description": "Balanced I2V - Controlled motion",
                "zones": "Comp(1) Mid(1) Detail(1)",
                "range": "0.500 → 0.000"
            },
            
            PresetCategory.STYLISED_MOTION.value: {
                "base": [0.7, 0.4, 0.2, 0.0],
                "description": "Stylised Motion - Artistic movement",
                "zones": "Comp(1) Mid(2) Detail(1)",
                "range": "0.700 → 0.000"
            },
            
            PresetCategory.MANUAL_PRESET.value: {
                "base": [0.909375, 0.725, 0.421875, 0.0],
                "description": "Manual Preset - Custom values",
                "zones": "Comp(1) Mid(2) Detail(1)",
                "range": "0.909 → 0.000"
            },
            
            PresetCategory.MID_SIGMA_FOCUS.value: {
                "base": [1.0, 0.6, 0.2, 0.0],
                "description": "Mid-Sigma Focus - Structure bias",
                "zones": "Comp(1) Mid(2) Detail(1)",
                "range": "1.000 → 0.000"
            },
            
            PresetCategory.HIGH_DETAIL_TAIL.value: {
                "base": [0.6, 0.3, 0.1, 0.0],
                "description": "High Detail Tail - Long detail refinement",
                "zones": "Comp(1) Mid(2) Detail(1)",
                "range": "0.600 → 0.000"
            },
            
            PresetCategory.EXPERIMENTAL_WIDE.value: {
                "base": [1.2, 0.8, 0.4, 0.0],
                "description": "Experimental Wide - Broad range",
                "zones": "Comp(1) Mid(2) Detail(1)",
                "range": "1.200 → 0.000"
            }
        }

    def _scale_sigmas_to_steps(self, base_sigmas, target_steps):
        """Scale a preset sigma schedule to match desired number of steps"""
        if len(base_sigmas) == target_steps:
            return base_sigmas.copy()
        
        # Use interpolation to scale the schedule
        x_original = np.linspace(0, 1, len(base_sigmas))
        x_target = np.linspace(0, 1, target_steps)
        
        scaled = np.interp(x_target, x_original, base_sigmas)
        
        # Ensure final sigma is 0
        if target_steps > 0:
            scaled[-1] = 0.0
            
        return scaled.tolist()

    def _generate_sigma_graph(self, sigmas, preset_name, preset_info):
        """Generate an interactive graph of the sigma schedule"""
        plt.figure(figsize=(8, 4))
        
        steps = list(range(len(sigmas)))
        
        # Plot sigma schedule
        plt.plot(steps, sigmas, 'b-o', linewidth=2, markersize=6, markerfacecolor='red')
        
        # Add step points as interactive markers
        for i, sigma in enumerate(sigmas):
            plt.plot(i, sigma, 'ro', markersize=8, picker=5, label='Adjustable Points' if i == 0 else "")
        
        plt.title(f"Sigma Schedule: {preset_name}\n{preset_info['description']}")
        plt.xlabel("Step")
        plt.ylabel("Sigma Value")
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, max(sigmas) * 1.1)
        
        # Add zone annotations based on preset info
        if "zones" in preset_info:
            zones = preset_info["zones"].split()
            plt.text(0.02, 0.98, f"Zones: {preset_info['zones']}", 
                    transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Convert to tensor for output
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor

    def get_sigmas(self, preset, steps, auto_scale_to_steps):
        """Main function to get sigmas based on preset"""
        
        # Get preset data
        preset_data = self.preset_sigmas.get(preset, self.preset_sigmas[PresetCategory.ZIMAGE.value])
        base_sigmas = preset_data["base"]
        
        # Scale if needed
        if auto_scale_to_steps and len(base_sigmas) != steps:
            sigmas_list = self._scale_sigmas_to_steps(base_sigmas, steps)
        else:
            sigmas_list = base_sigmas.copy()
        
        # Convert to tensor
        sigma_tensor = torch.tensor(sigmas_list, dtype=torch.float32)
        
        # Generate preset info string
        preset_info_str = json.dumps({
            "preset": preset,
            "description": preset_data["description"],
            "zones": preset_data["zones"],
            "range": preset_data["range"],
            "steps": len(sigmas_list),
            "original_steps": len(base_sigmas),
            "auto_scaled": auto_scale_to_steps and len(base_sigmas) != steps
        })
        
        # Generate graph
        graph_tensor = self._generate_sigma_graph(sigmas_list, preset, preset_data)
        
        return (sigma_tensor, graph_tensor, preset_info_str)


# Node registration
NODE_CLASS_MAPPINGS = {
    "GR Sigma Preset Selector": GRSigmaPresetSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GR Sigma Preset Selector": "GR Sigma Preset Selector"
}