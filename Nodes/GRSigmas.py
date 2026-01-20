import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from enum import Enum
from PIL import Image

class CurveType(Enum):
    LINEAR = "linear"
    EXP = "exp"
    LOG = "log"
    COSINE = "cosine"
    POLY = "poly"

class PresetType(Enum):
    CUSTOM = "custom"
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
    ZIMAGE = "zimage"  # Added ZImage preset

class GRSigmaPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "Ultra Lock (0.0 only) – No change, structure frozen",
                    "Micro Detail (0.1 → 0.0) – Texture polish only",
                    "Low Motion (0.2 → 0.0) – Subtle motion, strong lock",
                    "Img2Img Safe (0.3 → 0.0) – Classic stable img2img",
                    "Balanced I2V (0.5 → 0.0) – Controlled motion",
                    "Stylised Motion (0.7 → 0.0) – Artistic movement",
                    "Your Manual Preset (0.909375, 0.725, 0.421875, 0.0)",
                    "Mid-Sigma Focus (1.0 → 0.2 → 0.0) – Structure bias",
                    "High Detail Tail (0.6 → 0.3 → 0.1 → 0.0)",
                    "Experimental Wide (1.2 → 0.8 → 0.4 → 0.0)",
                ],)
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/sigmas"

    def get_sigmas(self, preset):
        presets = {
            "Ultra Lock (0.0 only) – No change, structure frozen":
                [0.0],

            "Micro Detail (0.1 → 0.0) – Texture polish only":
                [0.1, 0.0],

            "Low Motion (0.2 → 0.0) – Subtle motion, strong lock":
                [0.2, 0.0],

            "Img2Img Safe (0.3 → 0.0) – Classic stable img2img":
                [0.3, 0.0],

            "Balanced I2V (0.5 → 0.0) – Controlled motion":
                [0.5, 0.25, 0.0],

            "Stylised Motion (0.7 → 0.0) – Artistic movement":
                [0.7, 0.4, 0.2, 0.0],

            "Your Manual Preset (0.909375, 0.725, 0.421875, 0.0)":
                [0.909375, 0.725, 0.421875, 0.0],

            "Mid-Sigma Focus (1.0 → 0.2 → 0.0) – Structure bias":
                [1.0, 0.6, 0.2, 0.0],

            "High Detail Tail (0.6 → 0.3 → 0.1 → 0.0)":
                [0.6, 0.3, 0.1, 0.0],

            "Experimental Wide (1.2 → 0.8 → 0.4 → 0.0)":
                [1.2, 0.8, 0.4, 0.0],
        }

        sigmas = torch.tensor(presets[preset], dtype=torch.float32)
        return (sigmas,)


class GRSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        curve_types = [curve.value for curve in CurveType]
        preset_types = [preset.value for preset in PresetType]
        return {
            "required": {
                # Preset selection
                "preset": (preset_types, {"default": PresetType.CUSTOM.value}),
                
                # Global bounds
                "overall_max": ("FLOAT", {"default": 1.00, "min": 0.100, "max": 10.0, "step": 0.001}),
                "overall_min": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.200, "step": 0.001}),
                
                # Automatic step distribution
                "auto_distribute": ("BOOLEAN", {"default": False}),
                "total_steps": ("INT", {"default": 24, "min": 1, "max": 100}),
                
                # Zone thresholds (must descend: comp > mid > detail)
                "comp_thresh": ("FLOAT", {"default": 0.80, "min": 0.50, "max": 0.95, "step": 0.001}),
                "mid_thresh": ("FLOAT", {"default": 0.50, "min": 0.10, "max": 0.90, "step": 0.001}),
                
                # Per-zone controls
                "comp_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "comp_curve": (curve_types, {"default": CurveType.EXP.value}),
                "comp_exponent": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                "mid_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "mid_curve": (curve_types, {"default": CurveType.LINEAR.value}),
                "mid_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                "detail_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "detail_curve": (curve_types, {"default": CurveType.LOG.value}),
                "detail_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                # Smoothing
                "zone_transition": (["sharp", "smooth"], {"default": "sharp"}),
                
                # Debug controls
                "show_debug": ("BOOLEAN", {"default": False}),
                "show_ascii": ("BOOLEAN", {"default": True}),
                "show_graph": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sigmas_input": ("SIGMAS", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "IMAGE")
    RETURN_NAMES = ("sigmas", "graph")
    FUNCTION = "generate"
    CATEGORY = "GraftingRayman/Sigmas"
    OUTPUT_NODE = True

    def apply_preset(self, preset):
        presets = {
            PresetType.BALANCED.value: {
                "comp_thresh": 0.75,
                "mid_thresh": 0.45,
                "comp_steps": 8,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 2.0,
                "mid_steps": 10,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 6,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 1.0,
            },
            PresetType.COMPOSITION_HEAVY.value: {
                "comp_thresh": 0.85,
                "mid_thresh": 0.40,
                "comp_steps": 12,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 2.5,
                "mid_steps": 8,
                "mid_curve": CurveType.COSINE.value,
                "mid_exponent": 1.5,
                "detail_steps": 4,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.8,
            },
            PresetType.DETAIL_HEAVY.value: {
                "comp_thresh": 0.65,
                "mid_thresh": 0.35,
                "comp_steps": 4,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 1.5,
                "mid_steps": 8,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 12,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.5,
            },
            PresetType.AGGRESSIVE.value: {
                "comp_thresh": 0.90,
                "mid_thresh": 0.60,
                "comp_steps": 10,
                "comp_curve": CurveType.POLY.value,
                "comp_exponent": 3.0,
                "mid_steps": 8,
                "mid_curve": CurveType.EXP.value,
                "mid_exponent": 2.0,
                "detail_steps": 6,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 1.5,
            },
            PresetType.SUBTLE.value: {
                "comp_thresh": 0.70,
                "mid_thresh": 0.40,
                "comp_steps": 6,
                "comp_curve": CurveType.COSINE.value,
                "comp_exponent": 1.0,
                "mid_steps": 12,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 6,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.8,
            },
            PresetType.PORTRAIT.value: {
                "comp_thresh": 0.80,
                "mid_thresh": 0.40,
                "comp_steps": 10,
                "comp_curve": CurveType.COSINE.value,
                "comp_exponent": 1.5,
                "mid_steps": 10,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 4,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.7,
            },
            PresetType.LANDSCAPE.value: {
                "comp_thresh": 0.70,
                "mid_thresh": 0.35,
                "comp_steps": 6,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 1.8,
                "mid_steps": 12,
                "mid_curve": CurveType.COSINE.value,
                "mid_exponent": 1.2,
                "detail_steps": 6,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.5,
            },
            PresetType.ARCHITECTURE.value: {
                "comp_thresh": 0.85,
                "mid_thresh": 0.50,
                "comp_steps": 12,
                "comp_curve": CurveType.POLY.value,
                "comp_exponent": 2.5,
                "mid_steps": 8,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 4,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 1.5,
            },
            PresetType.ABSTRACT.value: {
                "comp_thresh": 0.65,
                "mid_thresh": 0.30,
                "comp_steps": 5,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 1.2,
                "mid_steps": 8,
                "mid_curve": CurveType.COSINE.value,
                "mid_exponent": 0.8,
                "detail_steps": 11,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.3,
            },
            PresetType.FINE_DETAIL.value: {
                "comp_thresh": 0.60,
                "mid_thresh": 0.25,
                "comp_steps": 4,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 1.0,
                "mid_steps": 8,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 12,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.2,
            },
            PresetType.FAST_DECAY.value: {
                "comp_thresh": 0.90,
                "mid_thresh": 0.60,
                "comp_steps": 4,
                "comp_curve": CurveType.POLY.value,
                "comp_exponent": 3.0,
                "mid_steps": 6,
                "mid_curve": CurveType.EXP.value,
                "mid_exponent": 2.0,
                "detail_steps": 14,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 1.0,
            },
            PresetType.SLOW_DECAY.value: {
                "comp_thresh": 0.70,
                "mid_thresh": 0.40,
                "comp_steps": 10,
                "comp_curve": CurveType.COSINE.value,
                "comp_exponent": 1.0,
                "mid_steps": 10,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 4,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.5,
            },
            PresetType.MID_CENTRIC.value: {
                "comp_thresh": 0.75,
                "mid_thresh": 0.35,
                "comp_steps": 6,
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 1.5,
                "mid_steps": 14,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 4,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.8,
            },
            PresetType.HIGH_CONTRAST.value: {
                "comp_thresh": 0.85,
                "mid_thresh": 0.50,
                "comp_steps": 8,
                "comp_curve": CurveType.POLY.value,
                "comp_exponent": 3.0,
                "mid_steps": 8,
                "mid_curve": CurveType.EXP.value,
                "mid_exponent": 2.0,
                "detail_steps": 8,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 1.5,
            },
            PresetType.LOW_CONTRAST.value: {
                "comp_thresh": 0.70,
                "mid_thresh": 0.40,
                "comp_steps": 8,
                "comp_curve": CurveType.COSINE.value,
                "comp_exponent": 1.0,
                "mid_steps": 10,
                "mid_curve": CurveType.LINEAR.value,
                "mid_exponent": 1.0,
                "detail_steps": 6,
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.5,
            },
            # ZIMAGE PRESET - tuned for Z-Image Turbo model
            PresetType.ZIMAGE.value: {
                "comp_thresh": 0.92,  # Very high threshold to match Z-Image's high initial sigma
                "mid_thresh": 0.75,   # Lower threshold for middle zone
                "comp_steps": 2,      # First zone: 0.991 to 0.92 (quick drop)
                "comp_curve": CurveType.EXP.value,
                "comp_exponent": 3.0, # Steep initial drop
                "mid_steps": 4,       # Second zone: 0.935 to 0.750
                "mid_curve": CurveType.POLY.value,
                "mid_exponent": 2.0,  # Moderate curve
                "detail_steps": 3,    # Third zone: 0.6582 to 0.2000
                "detail_curve": CurveType.LOG.value,
                "detail_exponent": 0.5, # Gentle finish
            }
        }
        return presets.get(preset, {})

    def auto_distribute_steps(self, total_steps, preset):
        distributions = {
            PresetType.BALANCED.value: (0.35, 0.40, 0.25),
            PresetType.COMPOSITION_HEAVY.value: (0.50, 0.30, 0.20),
            PresetType.DETAIL_HEAVY.value: (0.20, 0.30, 0.50),
            PresetType.AGGRESSIVE.value: (0.40, 0.35, 0.25),
            PresetType.SUBTLE.value: (0.25, 0.50, 0.25),
            PresetType.CUSTOM.value: (0.35, 0.40, 0.25),
            PresetType.PORTRAIT.value: (0.40, 0.40, 0.20),
            PresetType.LANDSCAPE.value: (0.25, 0.50, 0.25),
            PresetType.ARCHITECTURE.value: (0.50, 0.35, 0.15),
            PresetType.ABSTRACT.value: (0.20, 0.35, 0.45),
            PresetType.FINE_DETAIL.value: (0.15, 0.35, 0.50),
            PresetType.FAST_DECAY.value: (0.15, 0.25, 0.60),
            PresetType.SLOW_DECAY.value: (0.40, 0.40, 0.20),
            PresetType.MID_CENTRIC.value: (0.25, 0.60, 0.15),
            PresetType.HIGH_CONTRAST.value: (0.35, 0.35, 0.30),
            PresetType.LOW_CONTRAST.value: (0.35, 0.40, 0.25),
            PresetType.ZIMAGE.value: (0.22, 0.45, 0.33),  # 2/9, 4/9, 3/9 ratio for Z-Image's 9 steps
        }
        
        comp_ratio, mid_ratio, detail_ratio = distributions.get(preset, (0.35, 0.40, 0.25))
        
        comp_steps = max(1, round(total_steps * comp_ratio))
        mid_steps = max(1, round(total_steps * mid_ratio))
        detail_steps = max(1, total_steps - comp_steps - mid_steps)
        
        return comp_steps, mid_steps, detail_steps

    def validate_inputs(self, overall_max, overall_min, comp_thresh, mid_thresh):
        if overall_min >= overall_max:
            raise ValueError("overall_min must be less than overall_max")
        if comp_thresh <= mid_thresh:
            raise ValueError("comp_thresh must be greater than mid_thresh")
        if not (0 < overall_min < overall_max):
            raise ValueError("Invalid overall sigma range")
        if not (0 < mid_thresh < comp_thresh < 1):
            raise ValueError("Thresholds must be between 0 and 1 with comp > mid")

    def make_segment(self, steps, curve_type, zone_max, zone_min, exponent=1.0):
        x = np.linspace(0, 1, steps)
        
        if curve_type == CurveType.LINEAR.value:
            y = x
        elif curve_type == CurveType.EXP.value:
            y = np.exp(exponent * x) - 1
            y = y / y.max()
        elif curve_type == CurveType.LOG.value:
            y = np.log1p(x * exponent)
            y = y / y.max()
        elif curve_type == CurveType.COSINE.value:
            y = 1 - np.cos(x * np.pi/2)
            y = y ** exponent
        elif curve_type == CurveType.POLY.value:
            y = x ** exponent
        
        return zone_max - (zone_max - zone_min) * y

    def smooth_transition(self, sigmas, transition_points):
        for point in transition_points:
            if point == 0 or point >= len(sigmas) - 1:
                continue
            
            start = max(0, point - 1)
            end = min(len(sigmas), point + 2)
            
            for i in range(start, end):
                alpha = (i - start) / (end - start - 1)
                sigmas[i] = sigmas[start] * (1 - alpha) + sigmas[end-1] * alpha
        
        return sigmas

    def generate_graph(self, sigmas, transition_points, overall_max, overall_min):
        plt.figure(figsize=(10, 6))
        
        plt.plot(sigmas, 'b-', linewidth=2, label='Sigma Schedule')
        
        for i, point in enumerate(transition_points[:-1]):
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
        
        zones = ["Composition", "Mid", "Detail"]
        for i, (start, end) in enumerate(zip(transition_points[:-1], transition_points[1:])):
            mid_point = start + (end - start) / 2
            plt.text(mid_point, overall_max * 0.9, zones[i], 
                    ha='center', va='center', backgroundcolor='white')
        
        plt.title("GR Sigma Schedule")
        plt.xlabel("Step")
        plt.ylabel("Sigma Value")
        plt.grid(True, alpha=0.3)
        plt.ylim(overall_min * 0.9, overall_max * 1.05)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor

    def extract_params_from_sigmas(self, sigmas_input):
        if sigmas_input is None or len(sigmas_input) == 0:
            return None
            
        sigmas = sigmas_input.numpy()
        overall_max = float(sigmas[0])
        overall_min = float(sigmas[-1])
        
        # Try to detect zone boundaries by finding significant drops
        diffs = np.diff(sigmas)
        threshold = np.percentile(np.abs(diffs), 90)
        transition_indices = np.where(np.abs(diffs) > threshold)[0] + 1
        
        # Ensure we have exactly 2 transitions for 3 zones
        if len(transition_indices) >= 2:
            comp_steps = transition_indices[0]
            mid_steps = transition_indices[1] - transition_indices[0]
            detail_steps = len(sigmas) - transition_indices[1]
            
            # Estimate thresholds based on sigma values
            comp_thresh = float(sigmas[transition_indices[0]] / overall_max)
            mid_thresh = float(sigmas[transition_indices[1]] / overall_max)
            
            return {
                "overall_max": overall_max,
                "overall_min": overall_min,
                "comp_thresh": comp_thresh,
                "mid_thresh": mid_thresh,
                "comp_steps": comp_steps,
                "mid_steps": mid_steps,
                "detail_steps": detail_steps,
                "sigmas": sigmas
            }
        return None

    def generate_zimage_sigmas(self, total_steps):
        """Generate exact Z-Image Turbo sigmas for any number of steps"""
        if total_steps >= 9:
            sigmas1 = [0.991, 0.98, 0.92]
            sigmas2 = [0.935, 0.90, 0.875, 0.750, 0.0000]
            sigmas3 = [0.6582, 0.4556, 0.2000, 0.0000]
        elif total_steps == 8:
            sigmas1 = [0.991, 0.98, 0.92]
            sigmas2 = [0.935, 0.90, 0.875, 0.750, 0.0000]
            sigmas3 = [0.6582, 0.3019, 0.0000]
        elif total_steps == 7:
            sigmas1 = [0.991, 0.98, 0.92]
            sigmas2 = [0.9350, 0.8916, 0.7600, 0.0000]
            sigmas3 = [0.6582, 0.3019, 0.0000]
        elif total_steps == 6:
            sigmas1 = [0.991, 0.980, 0.920]
            sigmas2 = [0.942, 0.780, 0.000]
            sigmas3 = [0.6582, 0.3019, 0.0000]
        elif total_steps == 5:
            sigmas1 = [0.991, 0.980, 0.920]
            sigmas2 = [0.942, 0.780, 0.000]
            sigmas3 = [0.6200, 0.0000]
        elif total_steps <= 4:
            sigmas1 = [0.991, 0.980, 0.920]
            sigmas2 = [0.942, 0.000]
            sigmas3 = [0.790, 0.000]
        
        # Concatenate all sigmas and remove duplicates
        all_sigmas = sigmas1 + sigmas2 + sigmas3
        # Remove the extra 0.0000 at the end if it exists
        while len(all_sigmas) > total_steps + 1:
            all_sigmas.pop()
        
        return np.array(all_sigmas)

    def generate(self, preset, overall_max, overall_min, auto_distribute, total_steps,
                comp_thresh, mid_thresh, comp_steps, comp_curve, comp_exponent,
                mid_steps, mid_curve, mid_exponent, detail_steps, detail_curve, detail_exponent,
                zone_transition, show_debug, show_ascii, show_graph, sigmas_input=None):
        
        # If sigmas input is provided, extract parameters from it
        input_params = self.extract_params_from_sigmas(sigmas_input)
        if input_params is not None:
            # Use the parameters from the input sigmas
            overall_max = input_params["overall_max"]
            overall_min = input_params["overall_min"]
            comp_thresh = input_params["comp_thresh"]
            mid_thresh = input_params["mid_thresh"]
            comp_steps = input_params["comp_steps"]
            mid_steps = input_params["mid_steps"]
            detail_steps = input_params["detail_steps"]
            sigmas = input_params["sigmas"]
            
            # Generate transition points for visualization
            transition_points = [0, comp_steps, comp_steps + mid_steps, len(sigmas)]
            
            # Convert to tensor
            sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
        else:
            # Z-Image special handling - generate exact Z-Image sigmas
            if preset == PresetType.ZIMAGE.value:
                sigmas = self.generate_zimage_sigmas(total_steps)
                sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
                
                # Set up transition points based on Z-Image's 3-stage structure
                comp_steps = 3  # sigmas1 length
                mid_steps = 5   # sigmas2 length
                detail_steps = len(sigmas) - comp_steps - mid_steps + 1
                transition_points = [0, comp_steps, comp_steps + mid_steps - 1, len(sigmas)]
            else:
                # Proceed with normal generation if no sigmas input or couldn't parse
                if preset != PresetType.CUSTOM.value:
                    preset_params = self.apply_preset(preset)
                    comp_thresh = preset_params.get("comp_thresh", comp_thresh)
                    mid_thresh = preset_params.get("mid_thresh", mid_thresh)
                    comp_steps = preset_params.get("comp_steps", comp_steps)
                    comp_curve = preset_params.get("comp_curve", comp_curve)
                    comp_exponent = preset_params.get("comp_exponent", comp_exponent)
                    mid_steps = preset_params.get("mid_steps", mid_steps)
                    mid_curve = preset_params.get("mid_curve", mid_curve)
                    mid_exponent = preset_params.get("mid_exponent", mid_exponent)
                    detail_steps = preset_params.get("detail_steps", detail_steps)
                    detail_curve = preset_params.get("detail_curve", detail_curve)
                    detail_exponent = preset_params.get("detail_exponent", detail_exponent)
                
                if auto_distribute:
                    comp_steps, mid_steps, detail_steps = self.auto_distribute_steps(total_steps, preset)
                
                self.validate_inputs(overall_max, overall_min, comp_thresh, mid_thresh)
                
                comp_min = max(overall_max * comp_thresh, mid_thresh * 1.1)
                mid_min = max(overall_max * mid_thresh, overall_min * 1.1)
                
                comp_sig = self.make_segment(comp_steps, comp_curve, overall_max, comp_min, comp_exponent)
                mid_sig = self.make_segment(mid_steps, mid_curve, comp_min, mid_min, mid_exponent)
                detail_sig = self.make_segment(detail_steps, detail_curve, mid_min, overall_min, detail_exponent)
                
                sigmas = np.concatenate([comp_sig, mid_sig, detail_sig])
                transition_points = [0, comp_steps, comp_steps + mid_steps, len(sigmas)]
                
                if zone_transition == "smooth":
                    sigmas = self.smooth_transition(sigmas, transition_points)
                
                for i in range(1, len(sigmas)):
                    if sigmas[i] >= sigmas[i-1]:
                        sigmas[i] = sigmas[i-1] - 1e-6
                
                sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
        
        # Generate graph
        graph_tensor = self.generate_graph(sigmas, transition_points, overall_max, overall_min) if show_graph else torch.zeros((1, 1, 1, 3))
        
        # Debug output
        debug_text = self.generate_debug_output(
            sigma_tensor, overall_max, overall_min,
            comp_steps, mid_steps, detail_steps,
            comp_curve, mid_curve, detail_curve,
            show_debug, show_ascii, transition_points,
            preset, auto_distribute
        )

        return {
            "ui": {
                "text": [debug_text] if show_debug else ["Debug output disabled"],
            },
            "result": (sigma_tensor, graph_tensor)
        }

    def generate_debug_output(self, sigma_tensor, overall_max, overall_min,
                            comp_steps, mid_steps, detail_steps,
                            comp_curve, mid_curve, detail_curve,
                            show_debug, show_ascii, transition_points,
                            preset, auto_distribute):
        debug_text = ""
        
        if show_debug:
            debug_text = "=== SIGMA VALUES ===\n"
            debug_text += f"Preset: {preset}\n"
            debug_text += f"Auto Distribution: {'ON' if auto_distribute else 'OFF'}\n"
            debug_text += f"Total Steps: {len(sigma_tensor)}\n"
            debug_text += f"Global Range: {overall_max:.3f}→{overall_min:.3f}\n"
            debug_text += f"Zones: Comp({comp_steps}) Mid({mid_steps}) Detail({detail_steps})\n"
            if preset != PresetType.ZIMAGE.value:
                debug_text += f"Curves: Comp({comp_curve}) Mid({mid_curve}) Detail({detail_curve})\n"
            else:
                debug_text += "Curves: Z-Image Turbo (3-stage denoising)\n"
            debug_text += "----------------------------\n"
            debug_text += "\n".join([f"{i:02d}: {s:.6f}" for i, s in enumerate(sigma_tensor)])
        
        if show_ascii:
            self.print_ascii_visualization(
                sigma_tensor, overall_max, overall_min, 
                transition_points, preset, auto_distribute
            )
        
        return debug_text

    def print_ascii_visualization(self, sigma_tensor, overall_max, overall_min, 
                                transition_points, preset, auto_distribute):
        print("\n=== GR SIGMAS ===")
        print(f"Preset: {preset}")
        print(f"Auto Distribution: {'ON' if auto_distribute else 'OFF'}")
        print(f"Global Range: {overall_max:.2f}→{overall_min:.2f}")
        
        zone_names = ["COMPOSITION", "MID", "DETAIL"]
        zone_ranges = list(zip(transition_points[:-1], transition_points[1:]))
        
        for i, (start, end) in enumerate(zone_ranges):
            print(f"\n[{zone_names[i]}] Steps {start}-{end-1}")
            
            step_indices = range(start, min(end, start + 10))
            max_bar = 40
            
            for idx in step_indices:
                s = sigma_tensor[idx]
                bar = "■" * int(max_bar * (s - overall_min)/(overall_max - overall_min))
                diff = f"Δ={sigma_tensor[idx]-sigma_tensor[idx+1]:.3f}" if idx < len(sigma_tensor)-1 else ""
                print(f"{idx:02d}: {s:.4f} {diff}|{bar:<{max_bar}}|")
        
        print("\nValidation:")
        unique_vals = len(torch.unique(sigma_tensor))
        print(f"Unique Values: {unique_vals}/{len(sigma_tensor)}")
        print(f"Min Sigma: {sigma_tensor.min():.4f}")
        print(f"Max Sigma: {sigma_tensor.max():.4f}")
        print("=" * 60)


NODE_CLASS_MAPPINGS = {
    "GR Sigma Presets": GRSigmaPresets,
    "GR Sigmas": GRSigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GR Sigma Presets": "Sigma Presets (Manual)",
    "GR Sigmas": "GR Sigma Generator"
}
