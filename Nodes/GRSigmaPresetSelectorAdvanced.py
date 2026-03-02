import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from enum import Enum
from PIL import Image
import json

class PresetCategory(Enum):
    ZIMAGE            = "zimage"
    BALANCED          = "balanced"
    COMPOSITION_HEAVY = "composition_heavy"
    DETAIL_HEAVY      = "detail_heavy"
    AGGRESSIVE        = "aggressive"
    SUBTLE            = "subtle"
    PORTRAIT          = "portrait"
    LANDSCAPE         = "landscape"
    ARCHITECTURE      = "architecture"
    ABSTRACT          = "abstract"
    FINE_DETAIL       = "fine_detail"
    FAST_DECAY        = "fast_decay"
    SLOW_DECAY        = "slow_decay"
    MID_CENTRIC       = "mid_centric"
    HIGH_CONTRAST     = "high_contrast"
    LOW_CONTRAST      = "low_contrast"
    ULTRA_LOCK        = "ultra_lock"
    MICRO_DETAIL      = "micro_detail"
    LOW_MOTION        = "low_motion"
    IMG2IMG_SAFE      = "img2img_safe"
    BALANCED_I2V      = "balanced_i2v"
    STYLISED_MOTION   = "stylised_motion"
    MANUAL_PRESET     = "manual_preset"
    MID_SIGMA_FOCUS   = "mid_sigma_focus"
    HIGH_DETAIL_TAIL  = "high_detail_tail"
    EXPERIMENTAL_WIDE = "experimental_wide"


# ── Preset definitions — verbatim from GRSigmas.py apply_preset() ─────────────
PRESET_DEFS = {
    "balanced":          {"comp_thresh":0.75,"mid_thresh":0.45,"comp_steps":8, "comp_curve":"exp",   "comp_exp":2.0,"mid_steps":10,"mid_curve":"linear","mid_exp":1.0,"detail_steps":6, "detail_curve":"log","detail_exp":1.0},
    "composition_heavy": {"comp_thresh":0.85,"mid_thresh":0.40,"comp_steps":12,"comp_curve":"exp",   "comp_exp":2.5,"mid_steps":8, "mid_curve":"cosine","mid_exp":1.5,"detail_steps":4, "detail_curve":"log","detail_exp":0.8},
    "detail_heavy":      {"comp_thresh":0.65,"mid_thresh":0.35,"comp_steps":4, "comp_curve":"exp",   "comp_exp":1.5,"mid_steps":8, "mid_curve":"linear","mid_exp":1.0,"detail_steps":12,"detail_curve":"log","detail_exp":0.5},
    "aggressive":        {"comp_thresh":0.90,"mid_thresh":0.60,"comp_steps":10,"comp_curve":"poly",  "comp_exp":3.0,"mid_steps":8, "mid_curve":"exp",   "mid_exp":2.0,"detail_steps":6, "detail_curve":"log","detail_exp":1.5},
    "subtle":            {"comp_thresh":0.70,"mid_thresh":0.40,"comp_steps":6, "comp_curve":"cosine","comp_exp":1.0,"mid_steps":12,"mid_curve":"linear","mid_exp":1.0,"detail_steps":6, "detail_curve":"log","detail_exp":0.8},
    "portrait":          {"comp_thresh":0.80,"mid_thresh":0.40,"comp_steps":10,"comp_curve":"cosine","comp_exp":1.5,"mid_steps":10,"mid_curve":"linear","mid_exp":1.0,"detail_steps":4, "detail_curve":"log","detail_exp":0.7},
    "landscape":         {"comp_thresh":0.70,"mid_thresh":0.35,"comp_steps":6, "comp_curve":"exp",   "comp_exp":1.8,"mid_steps":12,"mid_curve":"cosine","mid_exp":1.2,"detail_steps":6, "detail_curve":"log","detail_exp":0.5},
    "architecture":      {"comp_thresh":0.85,"mid_thresh":0.50,"comp_steps":12,"comp_curve":"poly",  "comp_exp":2.5,"mid_steps":8, "mid_curve":"linear","mid_exp":1.0,"detail_steps":4, "detail_curve":"log","detail_exp":1.5},
    "abstract":          {"comp_thresh":0.65,"mid_thresh":0.30,"comp_steps":5, "comp_curve":"exp",   "comp_exp":1.2,"mid_steps":8, "mid_curve":"cosine","mid_exp":0.8,"detail_steps":11,"detail_curve":"log","detail_exp":0.3},
    "fine_detail":       {"comp_thresh":0.60,"mid_thresh":0.25,"comp_steps":4, "comp_curve":"exp",   "comp_exp":1.0,"mid_steps":8, "mid_curve":"linear","mid_exp":1.0,"detail_steps":12,"detail_curve":"log","detail_exp":0.2},
    "fast_decay":        {"comp_thresh":0.90,"mid_thresh":0.60,"comp_steps":4, "comp_curve":"poly",  "comp_exp":3.0,"mid_steps":6, "mid_curve":"exp",   "mid_exp":2.0,"detail_steps":14,"detail_curve":"log","detail_exp":1.0},
    "slow_decay":        {"comp_thresh":0.70,"mid_thresh":0.40,"comp_steps":10,"comp_curve":"cosine","comp_exp":1.0,"mid_steps":10,"mid_curve":"linear","mid_exp":1.0,"detail_steps":4, "detail_curve":"log","detail_exp":0.5},
    "mid_centric":       {"comp_thresh":0.75,"mid_thresh":0.35,"comp_steps":6, "comp_curve":"exp",   "comp_exp":1.5,"mid_steps":14,"mid_curve":"linear","mid_exp":1.0,"detail_steps":4, "detail_curve":"log","detail_exp":0.8},
    "high_contrast":     {"comp_thresh":0.85,"mid_thresh":0.50,"comp_steps":8, "comp_curve":"poly",  "comp_exp":3.0,"mid_steps":8, "mid_curve":"exp",   "mid_exp":2.0,"detail_steps":8, "detail_curve":"log","detail_exp":1.5},
    "low_contrast":      {"comp_thresh":0.70,"mid_thresh":0.40,"comp_steps":8, "comp_curve":"cosine","comp_exp":1.0,"mid_steps":10,"mid_curve":"linear","mid_exp":1.0,"detail_steps":6, "detail_curve":"log","detail_exp":0.5},
}

MANUAL_PRESETS = {
    "ultra_lock":        [0.0],
    "micro_detail":      [0.1, 0.0],
    "low_motion":        [0.2, 0.0],
    "img2img_safe":      [0.3, 0.0],
    "balanced_i2v":      [0.5, 0.25, 0.0],
    "stylised_motion":   [0.7, 0.4, 0.2, 0.0],
    "manual_preset":     [0.909375, 0.725, 0.421875, 0.0],
    "mid_sigma_focus":   [1.0, 0.6, 0.2, 0.0],
    "high_detail_tail":  [0.6, 0.3, 0.1, 0.0],
    "experimental_wide": [1.2, 0.8, 0.4, 0.0],
}

PRESET_DESCRIPTIONS = {
    "balanced":          "Balanced composition and detail",
    "composition_heavy": "Heavy composition focus",
    "detail_heavy":      "Heavy fine detail focus",
    "aggressive":        "Aggressive denoising",
    "subtle":            "Subtle smooth transitions",
    "portrait":          "Portrait optimised",
    "landscape":         "Landscape optimised",
    "architecture":      "Architecture optimised",
    "abstract":          "Abstract creative style",
    "fine_detail":       "Maximum fine detail",
    "fast_decay":        "Fast initial decay",
    "slow_decay":        "Slow gradual decay",
    "mid_centric":       "Middle stages focus",
    "high_contrast":     "High contrast transitions",
    "low_contrast":      "Low contrast smooth",
    "zimage":            "Z-Image Turbo exact schedule",
    "ultra_lock":        "Ultra Lock - structure frozen",
    "micro_detail":      "Micro Detail - texture polish",
    "low_motion":        "Low Motion - subtle, strong lock",
    "img2img_safe":      "Img2Img Safe - classic i2i",
    "balanced_i2v":      "Balanced I2V - controlled motion",
    "stylised_motion":   "Stylised Motion - artistic",
    "manual_preset":     "Manual Preset - custom values",
    "mid_sigma_focus":   "Mid-Sigma Focus - structure bias",
    "high_detail_tail":  "High Detail Tail - long refinement",
    "experimental_wide": "Experimental Wide - broad range",
}

# Zone step counts for graph annotation (comp, mid, detail)
PRESET_ZONE_STEPS = {
    "balanced":          (8,  10, 6),
    "composition_heavy": (12, 8,  4),
    "detail_heavy":      (4,  8,  12),
    "aggressive":        (10, 8,  6),
    "subtle":            (6,  12, 6),
    "portrait":          (10, 10, 4),
    "landscape":         (6,  12, 6),
    "architecture":      (12, 8,  4),
    "abstract":          (5,  8,  11),
    "fine_detail":       (4,  8,  12),
    "fast_decay":        (4,  6,  14),
    "slow_decay":        (10, 10, 4),
    "mid_centric":       (6,  14, 4),
    "high_contrast":     (8,  8,  8),
    "low_contrast":      (8,  10, 6),
    "zimage":            (3,  5,  3),
}


# ── Generation — exact port of GRSigmas.py ────────────────────────────────────

def make_segment(steps, curve_type, zone_max, zone_min, exponent=1.0):
    """Exact port of GRSigmas.py make_segment()"""
    x = np.linspace(0, 1, steps)
    if curve_type == "linear":
        y = x
    elif curve_type == "exp":
        y = np.exp(exponent * x) - 1
        y = y / y.max()
    elif curve_type == "log":
        y = np.log1p(x * exponent)
        y = y / y.max()
    elif curve_type == "cosine":
        y = 1 - np.cos(x * np.pi / 2)
        y = y ** exponent
    elif curve_type == "poly":
        y = x ** exponent
    else:
        y = x
    return zone_max - (zone_max - zone_min) * y


def generate_from_def(d, overall_max=1.0, overall_min=0.01):
    """Generate sigma list from a PRESET_DEFS entry, identical to GRSigmas.py generate()"""
    comp_min = max(overall_max * d["comp_thresh"], d["mid_thresh"] * 1.1)
    mid_min  = max(overall_max * d["mid_thresh"],  overall_min * 1.1)
    sigmas = np.concatenate([
        make_segment(d["comp_steps"],   d["comp_curve"],   overall_max, comp_min,    d["comp_exp"]),
        make_segment(d["mid_steps"],    d["mid_curve"],    comp_min,    mid_min,     d["mid_exp"]),
        make_segment(d["detail_steps"], d["detail_curve"], mid_min,     overall_min, d["detail_exp"]),
    ])
    for i in range(1, len(sigmas)):
        if sigmas[i] >= sigmas[i - 1]:
            sigmas[i] = sigmas[i - 1] - 1e-6
    return sigmas.tolist()


def generate_zimage_sigmas(total_steps=9):
    """Exact Z-Image Turbo sigmas from GRSigmas.py generate_zimage_sigmas()"""
    if total_steps >= 9:
        s = [0.991, 0.98, 0.92, 0.935, 0.90, 0.875, 0.750, 0.6582, 0.4556, 0.2000, 0.0000]
    elif total_steps == 8:
        s = [0.991, 0.98, 0.92, 0.935, 0.90, 0.875, 0.750, 0.6582, 0.3019, 0.0000]
    elif total_steps == 7:
        s = [0.991, 0.98, 0.92, 0.9350, 0.8916, 0.7600, 0.6582, 0.3019, 0.0000]
    elif total_steps == 6:
        s = [0.991, 0.980, 0.920, 0.942, 0.780, 0.6582, 0.3019, 0.0000]
    elif total_steps == 5:
        s = [0.991, 0.980, 0.920, 0.942, 0.780, 0.6200, 0.0000]
    else:
        s = [0.991, 0.980, 0.920, 0.942, 0.790, 0.0000]
    while len(s) > total_steps + 1:
        s.pop()
    return s


def get_sigmas_for_preset(preset):
    """Return dynamically generated sigma list for any preset key."""
    if preset == "zimage":
        return generate_zimage_sigmas(9)
    if preset in PRESET_DEFS:
        return generate_from_def(PRESET_DEFS[preset])
    if preset in MANUAL_PRESETS:
        return list(MANUAL_PRESETS[preset])
    return generate_from_def(PRESET_DEFS["balanced"])


def apply_curve_shape(sigmas, shape):
    """Remap sigmas through the selected curve shape, matching JS deriveDisplaySigmas()."""
    if shape == "preset" or len(sigmas) < 3:
        return sigmas
    fns = {
        "linear":  lambda t: t,
        "easein":  lambda t: t * t,
        "easeout": lambda t: 1 - (1 - t) ** 2,
        "cosine":  lambda t: (1 - np.cos(t * np.pi)) / 2,
        "s-curve": lambda t: 2*t*t if t < 0.5 else 1 - 2*(1-t)**2,
    }
    fn = fns.get(shape)
    if fn is None:
        return sigmas
    n = len(sigmas)
    start, end = sigmas[0], sigmas[-1]
    result = [start + (end - start) * fn(i / (n - 1)) for i in range(n - 1)]
    result.append(end)
    return result


# ── Node class ────────────────────────────────────────────────────────────────

class GRSigmaPresetSelectorAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = [p.value for p in PresetCategory]
        return {
            "required": {
                "preset": (preset_list, {"default": PresetCategory.ZIMAGE.value}),
                "steps":  ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "auto_scale_to_steps": ("BOOLEAN", {"default": True}),
                "curve_shape": (["preset", "linear", "easein", "easeout", "cosine", "s-curve"],
                                {"default": "preset"}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES = ("sigmas", "graph_preview", "preset_info")
    FUNCTION     = "get_sigmas"
    CATEGORY     = "GraftingRayman/Sigmas"
    OUTPUT_NODE  = True

    def _generate_graph(self, sigmas, preset):
        """Generate a styled graph image showing the true curve shape."""
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        n     = len(sigmas)
        steps = list(range(n))
        s_max = max(sigmas) if sigmas else 1.0

        # Zone shading + dividers + labels
        zone_steps = PRESET_ZONE_STEPS.get(preset)
        if zone_steps and n > 1:
            c, m, _ = zone_steps
            c_end = c - 1        # step index where comp ends
            m_end = c + m - 1    # step index where mid ends
            ax.axvspan(0,     c_end, alpha=0.08, color="#ef4444", zorder=0)
            ax.axvspan(c_end, m_end, alpha=0.08, color="#eab308", zorder=0)
            ax.axvspan(m_end, n - 1, alpha=0.08, color="#22c55e", zorder=0)
            ax.axvline(c_end, color="#fbbf24", linewidth=1, linestyle="--", alpha=0.55)
            ax.axvline(m_end, color="#4ade80",  linewidth=1, linestyle="--", alpha=0.55)
            ax.text(c_end / 2,            s_max * 1.06, "COMP",   color="#f87171", ha="center", fontsize=8, fontweight="bold")
            ax.text((c_end + m_end) / 2,  s_max * 1.06, "MID",    color="#fbbf24", ha="center", fontsize=8, fontweight="bold")
            ax.text((m_end + n - 1) / 2,  s_max * 1.06, "DETAIL", color="#4ade80", ha="center", fontsize=8, fontweight="bold")

        # Gradient fill under curve
        ax.fill_between(steps, sigmas, alpha=0.2, color="#3b82f6", zorder=1)

        # Curve line
        ax.plot(steps, sigmas, color="#3b82f6", linewidth=2, zorder=2)

        # Dots — last one grey (terminal sigma)
        ax.scatter(steps[:-1], sigmas[:-1], color="#3b82f6", s=45, zorder=3,
                   edgecolors="#0f172a", linewidths=1.2)
        ax.scatter([steps[-1]], [sigmas[-1]], color="#475569", s=45, zorder=3,
                   edgecolors="#0f172a", linewidths=1.2)

        # Grid
        ax.grid(True, color="#1e3a5f", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xlim(-0.4, n - 0.6)
        ax.set_ylim(-0.02, s_max * 1.14)

        ax.set_xlabel("Step",  color="#475569", fontsize=9)
        ax.set_ylabel("Sigma", color="#475569", fontsize=9)
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")

        desc = PRESET_DESCRIPTIONS.get(preset, preset)
        ax.set_title(f"{preset}  —  {desc}", color="#93c5fd", fontsize=10, pad=8)

        plt.tight_layout(pad=0.6)
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    def get_sigmas(self, preset, steps, auto_scale_to_steps, curve_shape="preset"):
        # Generate from the dynamic preset definition, then apply curve shape
        sigmas_list = apply_curve_shape(get_sigmas_for_preset(preset), curve_shape)

        sigma_tensor = torch.tensor(sigmas_list, dtype=torch.float32)
        graph_tensor = self._generate_graph(sigmas_list, preset)

        zone_steps = PRESET_ZONE_STEPS.get(preset, (0, 0, 0))
        c, m, d    = zone_steps

        preset_info_str = json.dumps({
            "preset":      preset,
            "description": PRESET_DESCRIPTIONS.get(preset, ""),
            "zones":       f"Comp({c}) Mid({m}) Detail({d})",
            "steps":       len(sigmas_list),
        })

        return (sigma_tensor, graph_tensor, preset_info_str)


# Node registration
NODE_CLASS_MAPPINGS = {
    "GR Sigma Preset Selector Advanced": GRSigmaPresetSelectorAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GR Sigma Preset Selector Advanced": "GR Sigma Preset Selector Advanced"
}