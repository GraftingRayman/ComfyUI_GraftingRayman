import { app } from "/scripts/app.js";

// ═══════════════════════════════════════════════════════════════════════════════
// SIGMA GENERATION — exact port of GRSigmas.py make_segment + generate logic
// ═══════════════════════════════════════════════════════════════════════════════

function makeSegment(steps, curveType, zoneMax, zoneMin, exponent = 1.0) {
    const x = Array.from({ length: steps }, (_, i) => i / (steps - 1 || 1));
    let y;
    switch (curveType) {
        case "exp":
            y = x.map(v => Math.exp(exponent * v) - 1);
            const yMaxExp = Math.max(...y, 1e-10);
            y = y.map(v => v / yMaxExp);
            break;
        case "log":
            y = x.map(v => Math.log1p(v * exponent));
            const yMaxLog = Math.max(...y, 1e-10);
            y = y.map(v => v / yMaxLog);
            break;
        case "cosine":
            y = x.map(v => Math.pow(1 - Math.cos(v * Math.PI / 2), exponent));
            break;
        case "poly":
            y = x.map(v => Math.pow(v, exponent));
            break;
        default: // linear
            y = x.map(v => v);
    }
    return y.map(v => zoneMax - (zoneMax - zoneMin) * v);
}

function enforceMonotonic(sigmas) {
    for (let i = 1; i < sigmas.length; i++) {
        if (sigmas[i] >= sigmas[i - 1]) sigmas[i] = sigmas[i - 1] - 1e-6;
    }
    return sigmas;
}

function generateFromDef(def, overallMax = 1.0, overallMin = 0.01) {
    const compMin = Math.max(overallMax * def.comp_thresh, def.mid_thresh * 1.1);
    const midMin  = Math.max(overallMax * def.mid_thresh,  overallMin * 1.1);
    return enforceMonotonic([
        ...makeSegment(def.comp_steps,   def.comp_curve,   overallMax, compMin,     def.comp_exponent),
        ...makeSegment(def.mid_steps,    def.mid_curve,    compMin,    midMin,      def.mid_exponent),
        ...makeSegment(def.detail_steps, def.detail_curve, midMin,     overallMin,  def.detail_exponent),
    ]);
}

function generateZImageSigmas(totalSteps) {
    let s;
    if      (totalSteps >= 9) s = [0.991,0.98,0.92,0.935,0.90,0.875,0.750,0.6582,0.4556,0.2000,0.0000];
    else if (totalSteps === 8) s = [0.991,0.98,0.92,0.935,0.90,0.875,0.750,0.6582,0.3019,0.0000];
    else if (totalSteps === 7) s = [0.991,0.98,0.92,0.9350,0.8916,0.7600,0.6582,0.3019,0.0000];
    else if (totalSteps === 6) s = [0.991,0.980,0.920,0.942,0.780,0.6582,0.3019,0.0000];
    else if (totalSteps === 5) s = [0.991,0.980,0.920,0.942,0.780,0.6200,0.0000];
    else                       s = [0.991,0.980,0.920,0.942,0.790,0.0000];
    while (s.length > totalSteps + 1) s.pop();
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRESET DEFINITIONS — verbatim from GRSigmas.py apply_preset()
// ═══════════════════════════════════════════════════════════════════════════════
const PRESET_DEFS = {
    balanced:          { comp_thresh:0.75, mid_thresh:0.45, comp_steps:8,  comp_curve:"exp",    comp_exponent:2.0, mid_steps:10, mid_curve:"linear", mid_exponent:1.0, detail_steps:6,  detail_curve:"log", detail_exponent:1.0 },
    composition_heavy: { comp_thresh:0.85, mid_thresh:0.40, comp_steps:12, comp_curve:"exp",    comp_exponent:2.5, mid_steps:8,  mid_curve:"cosine",  mid_exponent:1.5, detail_steps:4,  detail_curve:"log", detail_exponent:0.8 },
    detail_heavy:      { comp_thresh:0.65, mid_thresh:0.35, comp_steps:4,  comp_curve:"exp",    comp_exponent:1.5, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:12, detail_curve:"log", detail_exponent:0.5 },
    aggressive:        { comp_thresh:0.90, mid_thresh:0.60, comp_steps:10, comp_curve:"poly",   comp_exponent:3.0, mid_steps:8,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:6,  detail_curve:"log", detail_exponent:1.5 },
    subtle:            { comp_thresh:0.70, mid_thresh:0.40, comp_steps:6,  comp_curve:"cosine", comp_exponent:1.0, mid_steps:12, mid_curve:"linear",  mid_exponent:1.0, detail_steps:6,  detail_curve:"log", detail_exponent:0.8 },
    portrait:          { comp_thresh:0.80, mid_thresh:0.40, comp_steps:10, comp_curve:"cosine", comp_exponent:1.5, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log", detail_exponent:0.7 },
    landscape:         { comp_thresh:0.70, mid_thresh:0.35, comp_steps:6,  comp_curve:"exp",    comp_exponent:1.8, mid_steps:12, mid_curve:"cosine",  mid_exponent:1.2, detail_steps:6,  detail_curve:"log", detail_exponent:0.5 },
    architecture:      { comp_thresh:0.85, mid_thresh:0.50, comp_steps:12, comp_curve:"poly",   comp_exponent:2.5, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log", detail_exponent:1.5 },
    abstract:          { comp_thresh:0.65, mid_thresh:0.30, comp_steps:5,  comp_curve:"exp",    comp_exponent:1.2, mid_steps:8,  mid_curve:"cosine",  mid_exponent:0.8, detail_steps:11, detail_curve:"log", detail_exponent:0.3 },
    fine_detail:       { comp_thresh:0.60, mid_thresh:0.25, comp_steps:4,  comp_curve:"exp",    comp_exponent:1.0, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:12, detail_curve:"log", detail_exponent:0.2 },
    fast_decay:        { comp_thresh:0.90, mid_thresh:0.60, comp_steps:4,  comp_curve:"poly",   comp_exponent:3.0, mid_steps:6,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:14, detail_curve:"log", detail_exponent:1.0 },
    slow_decay:        { comp_thresh:0.70, mid_thresh:0.40, comp_steps:10, comp_curve:"cosine", comp_exponent:1.0, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log", detail_exponent:0.5 },
    mid_centric:       { comp_thresh:0.75, mid_thresh:0.35, comp_steps:6,  comp_curve:"exp",    comp_exponent:1.5, mid_steps:14, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log", detail_exponent:0.8 },
    high_contrast:     { comp_thresh:0.85, mid_thresh:0.50, comp_steps:8,  comp_curve:"poly",   comp_exponent:3.0, mid_steps:8,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:8,  detail_curve:"log", detail_exponent:1.5 },
    low_contrast:      { comp_thresh:0.70, mid_thresh:0.40, comp_steps:8,  comp_curve:"cosine", comp_exponent:1.0, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:6,  detail_curve:"log", detail_exponent:0.5 },
};

// Manual presets from GRSigmaPresets (GRSigmas.py)
const MANUAL_PRESETS = {
    ultra_lock:        [0.0],
    micro_detail:      [0.1, 0.0],
    low_motion:        [0.2, 0.0],
    img2img_safe:      [0.3, 0.0],
    balanced_i2v:      [0.5, 0.25, 0.0],
    stylised_motion:   [0.7, 0.4, 0.2, 0.0],
    manual_preset:     [0.909375, 0.725, 0.421875, 0.0],
    mid_sigma_focus:   [1.0, 0.6, 0.2, 0.0],
    high_detail_tail:  [0.6, 0.3, 0.1, 0.0],
    experimental_wide: [1.2, 0.8, 0.4, 0.0],
};

const PRESET_DESCRIPTIONS = {
    balanced:          "Balanced composition and detail",
    composition_heavy: "Heavy composition focus",
    detail_heavy:      "Heavy fine detail focus",
    aggressive:        "Aggressive denoising",
    subtle:            "Subtle smooth transitions",
    portrait:          "Portrait optimised",
    landscape:         "Landscape optimised",
    architecture:      "Architecture optimised",
    abstract:          "Abstract creative style",
    fine_detail:       "Maximum fine detail",
    fast_decay:        "Fast initial decay",
    slow_decay:        "Slow gradual decay",
    mid_centric:       "Middle stages focus",
    high_contrast:     "High contrast transitions",
    low_contrast:      "Low contrast smooth",
    zimage:            "Z-Image Turbo exact schedule",
    ultra_lock:        "Ultra Lock – structure frozen",
    micro_detail:      "Micro Detail – texture polish",
    low_motion:        "Low Motion – subtle, strong lock",
    img2img_safe:      "Img2Img Safe – classic i2i",
    balanced_i2v:      "Balanced I2V – controlled motion",
    stylised_motion:   "Stylised Motion – artistic",
    manual_preset:     "Manual Preset – custom values",
    mid_sigma_focus:   "Mid-Sigma Focus – structure bias",
    high_detail_tail:  "High Detail Tail – long refinement",
    experimental_wide: "Experimental Wide – broad range",
};

// Zone step counts for divider initialisation
const PRESET_ZONE_STEPS = {
    balanced:          { comp:8,  mid:10, detail:6  },
    composition_heavy: { comp:12, mid:8,  detail:4  },
    detail_heavy:      { comp:4,  mid:8,  detail:12 },
    aggressive:        { comp:10, mid:8,  detail:6  },
    subtle:            { comp:6,  mid:12, detail:6  },
    portrait:          { comp:10, mid:10, detail:4  },
    landscape:         { comp:6,  mid:12, detail:6  },
    architecture:      { comp:12, mid:8,  detail:4  },
    abstract:          { comp:5,  mid:8,  detail:11 },
    fine_detail:       { comp:4,  mid:8,  detail:12 },
    fast_decay:        { comp:4,  mid:6,  detail:14 },
    slow_decay:        { comp:10, mid:10, detail:4  },
    mid_centric:       { comp:6,  mid:14, detail:4  },
    high_contrast:     { comp:8,  mid:8,  detail:8  },
    low_contrast:      { comp:8,  mid:10, detail:6  },
    zimage:            { comp:3,  mid:5,  detail:3  },
};

function getSigmas(presetKey) {
    if (presetKey === "zimage")          return generateZImageSigmas(9);
    if (PRESET_DEFS[presetKey])          return generateFromDef(PRESET_DEFS[presetKey]);
    if (MANUAL_PRESETS[presetKey])       return [...MANUAL_PRESETS[presetKey]];
    return generateFromDef(PRESET_DEFS.balanced);
}

function zonesToFractions(presetKey, n) {
    const z = PRESET_ZONE_STEPS[presetKey];
    if (!z || n < 2) return [0.33, 0.66];
    const total = z.comp + z.mid + z.detail;
    return [z.comp / total, (z.comp + z.mid) / total];
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZONE COLORS
// ═══════════════════════════════════════════════════════════════════════════════
const ZONE_COLORS = {
    comp:   { fill: "rgba(239,68,68,0.07)",  line: "rgba(239,68,68,0.7)",  label: "#f87171" },
    mid:    { fill: "rgba(234,179,8,0.07)",  line: "rgba(234,179,8,0.7)",  label: "#fbbf24" },
    detail: { fill: "rgba(34,197,94,0.07)",  line: "rgba(34,197,94,0.7)",  label: "#4ade80" },
};

// ═══════════════════════════════════════════════════════════════════════════════
// LAYOUT
// ═══════════════════════════════════════════════════════════════════════════════
const GRAPH_H         = 170;
const GRAPH_PAD       = 6;
const PAD             = { top: 28, right: 10, bottom: 26, left: 36 };
const HANDLE_Y_OFFSET = 8;

function graphLayout(node, graphY, sigmas) {
    const gx = GRAPH_PAD;
    const gy = graphY;
    const gw = node.size[0] - GRAPH_PAD * 2;
    const gh = GRAPH_H;
    const px = gx + PAD.left,  py = gy + PAD.top;
    const pw = gw - PAD.left - PAD.right;
    const ph = gh - PAD.top  - PAD.bottom;
    const n  = sigmas.length;
    const maxSigma = Math.max(...sigmas, 0.01);

    const dotX       = i => px + (n > 1 ? (i / (n - 1)) * pw : pw / 2);
    const dotY       = s => py + ph - (s / maxSigma) * ph;
    const sigmaFromY = y => Math.max(0, Math.min(maxSigma, ((py + ph - y) / ph) * maxSigma));
    const divX       = f => px + f * pw;
    const handleY    = gy + HANDLE_Y_OFFSET;

    return { gx, gy, gw, gh, px, py, pw, ph, maxSigma, dotX, dotY, sigmaFromY, divX, handleY };
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAW
// ═══════════════════════════════════════════════════════════════════════════════
function drawGraph(ctx, node, graphY, sigmas, presetKey, hoveredDot, draggedDot, dividers, hoveredDiv, draggedDiv) {
    const L = graphLayout(node, graphY, sigmas);
    const { gx, gy, gw, gh, px, py, pw, ph, maxSigma, dotX, dotY, divX, handleY } = L;
    const n = sigmas.length;

    // Background + border
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(gx, gy, gw, gh);
    ctx.strokeStyle = "#1e3a5f"; ctx.lineWidth = 1;
    ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);

    // Zone fills
    for (const [x0, x1, col] of [
        [px,               divX(dividers[0]), ZONE_COLORS.comp],
        [divX(dividers[0]),divX(dividers[1]), ZONE_COLORS.mid],
        [divX(dividers[1]),px + pw,           ZONE_COLORS.detail],
    ]) {
        if (x1 > x0) { ctx.fillStyle = col.fill; ctx.fillRect(x0, py, x1 - x0, ph); }
    }

    // Grid + Y labels
    for (let g = 0; g <= 4; g++) {
        const lineY = py + (g / 4) * ph;
        ctx.strokeStyle = "#1e3a5f"; ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(px, lineY); ctx.lineTo(px + pw, lineY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#475569"; ctx.font = "9px monospace"; ctx.textAlign = "right";
        ctx.fillText((maxSigma * (1 - g / 4)).toFixed(3), px - 3, lineY + 3);
    }
    ctx.textAlign = "left";

    // Divider lines + diamond handles
    for (let d = 0; d < 2; d++) {
        const x   = divX(dividers[d]);
        const col = d === 0 ? ZONE_COLORS.mid.line : ZONE_COLORS.detail.line;
        const hot = draggedDiv === d || hoveredDiv === d;
        ctx.strokeStyle = hot ? col : col.replace("0.7", "0.4");
        ctx.lineWidth = hot ? 2 : 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x, py + ph); ctx.stroke();
        ctx.setLineDash([]);

        const hs = draggedDiv === d ? 7 : hoveredDiv === d ? 6 : 5;
        const hy = handleY;
        ctx.beginPath();
        ctx.moveTo(x, hy - hs); ctx.lineTo(x + hs, hy);
        ctx.lineTo(x, hy + hs); ctx.lineTo(x - hs, hy);
        ctx.closePath();
        ctx.fillStyle = hot ? col : col.replace("0.7", "0.45");
        ctx.fill();
        ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1; ctx.stroke();
    }

    // Zone labels
    for (const [x0, x1, col, lbl] of [
        [px,               divX(dividers[0]), ZONE_COLORS.comp.label,   "COMP"],
        [divX(dividers[0]),divX(dividers[1]), ZONE_COLORS.mid.label,    "MID"],
        [divX(dividers[1]),px + pw,           ZONE_COLORS.detail.label, "DETAIL"],
    ]) {
        if (x1 - x0 < 20) continue;
        ctx.fillStyle = col; ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
        ctx.fillText(lbl, (x0 + x1) / 2, gy + PAD.top - 4);
    }
    ctx.textAlign = "left";

    // Gradient fill under curve
    if (n > 1) {
        ctx.beginPath();
        ctx.moveTo(dotX(0), py + ph);
        for (let i = 0; i < n; i++) ctx.lineTo(dotX(i), dotY(sigmas[i]));
        ctx.lineTo(dotX(n - 1), py + ph);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, py, 0, py + ph);
        grad.addColorStop(0, "rgba(59,130,246,0.35)");
        grad.addColorStop(1, "rgba(59,130,246,0.03)");
        ctx.fillStyle = grad; ctx.fill();
    }

    // Curve line
    ctx.beginPath(); ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {
        i === 0 ? ctx.moveTo(dotX(i), dotY(sigmas[i])) : ctx.lineTo(dotX(i), dotY(sigmas[i]));
    }
    ctx.stroke();

    // Dots
    for (let i = 0; i < n; i++) {
        const dx = dotX(i), dy = dotY(sigmas[i]);
        const isDrag = draggedDot === i, isHov = hoveredDot === i, isLast = i === n - 1;
        ctx.beginPath();
        ctx.arc(dx, dy, isDrag ? 7 : isHov ? 6 : 5, 0, Math.PI * 2);
        ctx.fillStyle = isDrag ? "#f87171" : isHov ? "#93c5fd" : isLast ? "#64748b" : "#3b82f6";
        ctx.fill();
        ctx.strokeStyle = isDrag ? "#fca5a5" : "#0f172a"; ctx.lineWidth = 1.5; ctx.stroke();

        if (isHov || isDrag) {
            const label = sigmas[i].toFixed(4);
            ctx.font = "bold 10px monospace";
            const tw = ctx.measureText(label).width;
            const lx = Math.max(px, Math.min(dx - tw / 2, px + pw - tw - 2));
            const ly = dy - 11;
            ctx.fillStyle = "rgba(15,23,42,0.9)";
            ctx.fillRect(lx - 2, ly - 10, tw + 4, 13);
            ctx.fillStyle = "#e2e8f0"; ctx.textAlign = "left";
            ctx.fillText(label, lx, ly);
        }
    }
    ctx.textAlign = "left";

    // X step labels
    const every = Math.max(1, Math.floor(n / 8));
    ctx.fillStyle = "#475569"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    for (let i = 0; i < n; i += every) ctx.fillText(i, dotX(i), py + ph + 12);
    ctx.textAlign = "left";

    // Header: preset name + live zone counts
    ctx.fillStyle = "#93c5fd"; ctx.font = "bold 10px Arial";
    ctx.fillText(presetKey, gx + 4, gy + 12);
    const c   = Math.round(dividers[0] * (n - 1));
    const m   = Math.round((dividers[1] - dividers[0]) * (n - 1));
    const det = (n - 1) - c - m;
    ctx.fillStyle = "#475569"; ctx.font = "9px Arial"; ctx.textAlign = "right";
    ctx.fillText(`C:${c} M:${m} D:${det}`, gx + gw - 4, gy + 12);
    ctx.textAlign = "left";

    // Footer
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(gx, gy + gh - 14, gw, 14);
    let footerText, footerCol;
    if (draggedDot >= 0) {
        footerText = `step ${draggedDot}  σ = ${sigmas[draggedDot].toFixed(5)}  (right-click resets dot)`;
        footerCol  = "#fbbf24";
    } else if (draggedDiv >= 0) {
        footerText = `${draggedDiv === 0 ? "Comp|Mid" : "Mid|Detail"} divider  (drag to resize zones)`;
        footerCol  = draggedDiv === 0 ? ZONE_COLORS.mid.label : ZONE_COLORS.detail.label;
    } else {
        footerText = PRESET_DESCRIPTIONS[presetKey] || "";
        footerCol  = "#64748b";
    }
    ctx.fillStyle = footerCol; ctx.font = "9px Arial";
    ctx.fillText(footerText, gx + 4, gy + gh - 3);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTENSION
// ═══════════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "GraftingRayman.GRSigmaPresetSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GR Sigma Preset Selector") return;

        const DOT_HIT_R = 8;
        const DIV_HIT_R = 8;

        nodeType.prototype._graphY = function () {
            let bottom = 0;
            if (this.widgets) for (const w of this.widgets) {
                const y = (w.last_y ?? 0) + 24;
                if (y > bottom) bottom = y;
            }
            return bottom + GRAPH_PAD;
        };

        nodeType.prototype._widgetSig = function () {
            const p = this.widgets?.find(w => w.name === "preset");
            return `${p?.value}`;
        };

        nodeType.prototype._freshSigmas = function () {
            const p = this.widgets?.find(w => w.name === "preset");
            return getSigmas(p?.value ?? "balanced");
        };

        nodeType.prototype._freshDividers = function () {
            const p = this.widgets?.find(w => w.name === "preset");
            return zonesToFractions(p?.value ?? "balanced", this._sigmas?.length ?? 24);
        };

        nodeType.prototype._syncFromWidgets = function () {
            const sig = this._widgetSig();
            if (sig !== this._lastPresetSig) {
                this._lastPresetSig = sig;
                this._sigmas   = this._freshSigmas();
                this._dividers = this._freshDividers();
                this._hoveredDot = -1;
                this._hoveredDiv = -1;
            }
        };

        // ── lifecycle ────────────────────────────────────────────────────────

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            this._sigmas        = [];
            this._dividers      = [0.33, 0.66];
            this._draggedDot    = -1;
            this._hoveredDot    = -1;
            this._draggedDiv    = -1;
            this._hoveredDiv    = -1;
            this._lastPresetSig = null;
            setTimeout(() => {
                this._sigmas        = this._freshSigmas();
                this._dividers      = this._freshDividers();
                this._lastPresetSig = this._widgetSig();
                this.setDirtyCanvas(true, true);
            }, 60);
            return r;
        };

        // ── draw ─────────────────────────────────────────────────────────────

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (onDrawForeground) onDrawForeground.apply(this, arguments);
            this._syncFromWidgets();
            if (!this._sigmas?.length) return;
            const gy = this._graphY();
            if (this.size[1] < gy + GRAPH_H + GRAPH_PAD) this.size[1] = gy + GRAPH_H + GRAPH_PAD;
            const pk = this.widgets?.find(w => w.name === "preset")?.value ?? "balanced";
            drawGraph(ctx, this, gy, this._sigmas, pk,
                this._hoveredDot, this._draggedDot,
                this._dividers,   this._hoveredDiv, this._draggedDiv);
        };

        // ── mouse ─────────────────────────────────────────────────────────────

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos) {
            if (!this._sigmas?.length) return onMouseDown?.apply(this, arguments);
            const gy = this._graphY();
            const L  = graphLayout(this, gy, this._sigmas);

            for (let d = 0; d < 2; d++) {
                if (Math.abs(pos[0] - L.divX(this._dividers[d])) <= DIV_HIT_R &&
                    Math.abs(pos[1] - L.handleY) <= DIV_HIT_R) {
                    this._draggedDiv = d; this._hoveredDiv = d;
                    this.setDirtyCanvas(true, true); return true;
                }
            }

            const n = this._sigmas.length;
            for (let i = 0; i < n; i++) {
                if (Math.hypot(L.dotX(i) - pos[0], L.dotY(this._sigmas[i]) - pos[1]) <= DOT_HIT_R) {
                    if (e.button === 2) {
                        const fresh = this._freshSigmas();
                        if (i < fresh.length) this._sigmas[i] = fresh[i];
                        this.setDirtyCanvas(true, true); return true;
                    }
                    this._draggedDot = i; this._hoveredDot = i;
                    this.setDirtyCanvas(true, true); return true;
                }
            }
            return onMouseDown?.apply(this, arguments);
        };

        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, pos) {
            if (!this._sigmas?.length) return onMouseMove?.apply(this, arguments);
            const gy = this._graphY();
            const L  = graphLayout(this, gy, this._sigmas);
            const n  = this._sigmas.length;

            if (this._draggedDiv >= 0) {
                const d   = this._draggedDiv;
                const gap = n > 1 ? 1 / (n - 1) : 0.05;
                let frac  = (pos[0] - L.px) / L.pw;
                frac = d === 0
                    ? Math.max(gap, Math.min(this._dividers[1] - gap, frac))
                    : Math.max(this._dividers[0] + gap, Math.min(1 - gap, frac));
                this._dividers[d] = frac;
                document.body.style.cursor = "ew-resize";
                this.setDirtyCanvas(true, true); return true;
            }

            if (this._draggedDot >= 0) {
                const i = this._draggedDot;
                let s = L.sigmaFromY(pos[1]);
                if (i === n - 1) s = 0;
                else {
                    if (i > 0)   s = Math.min(s, this._sigmas[i - 1]);
                    if (i < n-1) s = Math.max(s, this._sigmas[i + 1]);
                }
                this._sigmas[i] = Math.round(s * 100000) / 100000;
                document.body.style.cursor = "ns-resize";
                this.setDirtyCanvas(true, true); return true;
            }

            let newHovDiv = -1;
            for (let d = 0; d < 2; d++) {
                if (Math.abs(pos[0] - L.divX(this._dividers[d])) <= DIV_HIT_R &&
                    Math.abs(pos[1] - L.handleY) <= DIV_HIT_R) { newHovDiv = d; break; }
            }
            let newHovDot = -1;
            if (newHovDiv < 0) {
                for (let i = 0; i < n; i++) {
                    if (Math.hypot(L.dotX(i) - pos[0], L.dotY(this._sigmas[i]) - pos[1]) <= DOT_HIT_R) {
                        newHovDot = i; break;
                    }
                }
            }
            if (newHovDiv !== this._hoveredDiv || newHovDot !== this._hoveredDot) {
                this._hoveredDiv = newHovDiv; this._hoveredDot = newHovDot;
                document.body.style.cursor = newHovDiv >= 0 ? "ew-resize" : newHovDot >= 0 ? "ns-resize" : "";
                this.setDirtyCanvas(true, true);
            }
            return onMouseMove?.apply(this, arguments);
        };

        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e) {
            if (this._draggedDiv >= 0 || this._draggedDot >= 0) {
                this._draggedDiv = -1; this._draggedDot = -1;
                document.body.style.cursor = "";
                this.setDirtyCanvas(true, true); return true;
            }
            return onMouseUp?.apply(this, arguments);
        };

        const onMouseLeave = nodeType.prototype.onMouseLeave;
        nodeType.prototype.onMouseLeave = function () {
            this._hoveredDot = this._hoveredDiv = this._draggedDot = this._draggedDiv = -1;
            document.body.style.cursor = "";
            this.setDirtyCanvas(true, true);
            return onMouseLeave?.apply(this, arguments);
        };

        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = computeSize?.apply(this, arguments) ?? [240, 120];
            const need = (this._graphY?.() ?? size[1]) + GRAPH_H + GRAPH_PAD;
            if (size[1] < need) size[1] = need;
            return size;
        };
    }
});