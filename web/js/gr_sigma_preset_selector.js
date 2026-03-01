import { app } from "/scripts/app.js";

// ─── Preset data ─────────────────────────────────────────────────────────────
const PRESETS = {
    zimage:            { base: [0.991,0.98,0.92,0.935,0.90,0.875,0.750,0.6582,0.4556,0.2000,0.0000], description: "Z-Image Turbo optimised schedule", zones: "Comp(3) Mid(5) Detail(3)" },
    flux:              { base: [1.00,0.95,0.85,0.72,0.58,0.45,0.32,0.20,0.10,0.05,0.00],             description: "Flux smooth decay",               zones: "Comp(4) Mid(4) Detail(3)" },
    flux_klein:        { base: [0.99,0.92,0.78,0.65,0.50,0.38,0.27,0.18,0.10,0.04,0.00],             description: "Flux Klein faster initial decay",  zones: "Comp(3) Mid(4) Detail(4)" },
    qwen:              { base: [0.98,0.94,0.86,0.75,0.62,0.48,0.35,0.24,0.15,0.07,0.00],             description: "QWEN balanced structure/detail",   zones: "Comp(4) Mid(4) Detail(3)" },
    sdxl:              { base: [0.95,0.88,0.78,0.65,0.52,0.40,0.30,0.22,0.15,0.08,0.00],             description: "SDXL standard schedule",           zones: "Comp(3) Mid(4) Detail(4)" },
    pony:              { base: [0.96,0.90,0.80,0.68,0.55,0.42,0.31,0.22,0.14,0.06,0.00],             description: "Pony gentle transitions",          zones: "Comp(3) Mid(4) Detail(4)" },
    sd:                { base: [0.94,0.86,0.74,0.60,0.48,0.36,0.26,0.18,0.11,0.05,0.00],             description: "SD 1.5/2.1 standard",             zones: "Comp(3) Mid(4) Detail(4)" },
    balanced:          { base: [1.0,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05,0.00],              description: "Balanced composition and detail",  zones: "Comp(4) Mid(4) Detail(3)" },
    composition_heavy: { base: [1.0,0.92,0.85,0.78,0.70,0.55,0.40,0.25,0.12,0.04,0.00],              description: "Heavy composition focus",          zones: "Comp(5) Mid(3) Detail(3)" },
    detail_heavy:      { base: [0.98,0.85,0.72,0.60,0.48,0.38,0.30,0.23,0.16,0.08,0.00],             description: "Heavy fine detail focus",          zones: "Comp(2) Mid(4) Detail(5)" },
    aggressive:        { base: [1.0,0.88,0.76,0.64,0.52,0.40,0.30,0.20,0.12,0.05,0.00],              description: "Aggressive denoising",             zones: "Comp(3) Mid(4) Detail(4)" },
    subtle:            { base: [0.95,0.86,0.78,0.70,0.62,0.54,0.44,0.34,0.22,0.10,0.00],             description: "Subtle smooth transitions",        zones: "Comp(3) Mid(4) Detail(4)" },
    portrait:          { base: [0.96,0.88,0.80,0.72,0.62,0.50,0.38,0.26,0.16,0.07,0.00],             description: "Portrait optimised",              zones: "Comp(3) Mid(4) Detail(4)" },
    landscape:         { base: [0.98,0.92,0.84,0.74,0.62,0.48,0.36,0.26,0.18,0.09,0.00],             description: "Landscape optimised",             zones: "Comp(3) Mid(4) Detail(4)" },
    architecture:      { base: [0.99,0.93,0.85,0.75,0.63,0.51,0.39,0.28,0.18,0.08,0.00],             description: "Architecture optimised",          zones: "Comp(4) Mid(4) Detail(3)" },
    abstract:          { base: [0.97,0.89,0.81,0.73,0.65,0.55,0.43,0.31,0.19,0.08,0.00],             description: "Abstract creative style",         zones: "Comp(3) Mid(4) Detail(4)" },
    fine_detail:       { base: [0.93,0.84,0.73,0.62,0.51,0.40,0.30,0.22,0.15,0.07,0.00],             description: "Maximum fine detail",             zones: "Comp(2) Mid(4) Detail(5)" },
    fast_decay:        { base: [1.0,0.90,0.78,0.64,0.50,0.36,0.24,0.14,0.07,0.02,0.00],              description: "Fast initial decay",              zones: "Comp(2) Mid(4) Detail(5)" },
    slow_decay:        { base: [0.96,0.91,0.86,0.80,0.72,0.62,0.50,0.38,0.24,0.10,0.00],             description: "Slow gradual decay",              zones: "Comp(4) Mid(4) Detail(3)" },
    mid_centric:       { base: [0.98,0.90,0.80,0.68,0.56,0.44,0.34,0.24,0.16,0.08,0.00],             description: "Middle stages focus",             zones: "Comp(3) Mid(5) Detail(3)" },
    high_contrast:     { base: [1.0,0.92,0.82,0.70,0.58,0.46,0.34,0.22,0.12,0.04,0.00],              description: "High contrast transitions",       zones: "Comp(3) Mid(4) Detail(4)" },
    low_contrast:      { base: [0.94,0.88,0.82,0.76,0.68,0.58,0.46,0.34,0.22,0.10,0.00],             description: "Low contrast smooth",             zones: "Comp(3) Mid(4) Detail(4)" },
    ultra_lock:        { base: [0.0],                                                                  description: "Ultra Lock – structure frozen",    zones: "Comp(1) Mid(0) Detail(0)" },
    micro_detail:      { base: [0.1,0.0],                                                              description: "Micro Detail – texture polish",    zones: "Comp(1) Mid(0) Detail(1)" },
    low_motion:        { base: [0.2,0.0],                                                              description: "Low Motion – subtle, strong lock", zones: "Comp(1) Mid(0) Detail(1)" },
    img2img_safe:      { base: [0.3,0.0],                                                              description: "Img2Img Safe – classic i2i",       zones: "Comp(1) Mid(0) Detail(1)" },
    balanced_i2v:      { base: [0.5,0.25,0.0],                                                         description: "Balanced I2V – controlled motion",  zones: "Comp(1) Mid(1) Detail(1)" },
    stylised_motion:   { base: [0.7,0.4,0.2,0.0],                                                      description: "Stylised Motion – artistic",       zones: "Comp(1) Mid(2) Detail(1)" },
    manual_preset:     { base: [0.909375,0.725,0.421875,0.0],                                          description: "Manual Preset – custom values",    zones: "Comp(1) Mid(2) Detail(1)" },
    mid_sigma_focus:   { base: [1.0,0.6,0.2,0.0],                                                      description: "Mid-Sigma Focus – structure bias", zones: "Comp(1) Mid(2) Detail(1)" },
    high_detail_tail:  { base: [0.6,0.3,0.1,0.0],                                                      description: "High Detail Tail – long refinement",zones: "Comp(1) Mid(2) Detail(1)" },
    experimental_wide: { base: [1.2,0.8,0.4,0.0],                                                      description: "Experimental Wide – broad range",  zones: "Comp(1) Mid(2) Detail(1)" },
};

// Zone colors
const ZONE_COLORS = {
    comp:   { fill: "rgba(239,68,68,0.07)",   line: "rgba(239,68,68,0.7)",   label: "#f87171" },  // red
    mid:    { fill: "rgba(234,179,8,0.07)",   line: "rgba(234,179,8,0.7)",   label: "#fbbf24" },  // yellow
    detail: { fill: "rgba(34,197,94,0.07)",   line: "rgba(34,197,94,0.7)",   label: "#4ade80" },  // green
};

// ─── Parse zones string → { comp, mid, detail } step counts ──────────────────
// e.g. "Comp(3) Mid(5) Detail(3)" → { comp:3, mid:5, detail:3 }
function parseZones(zonesStr) {
    const m = zonesStr?.match(/Comp\((\d+)\).*?Mid\((\d+)\).*?Detail\((\d+)\)/);
    if (!m) return null;
    return { comp: parseInt(m[1]), mid: parseInt(m[2]), detail: parseInt(m[3]) };
}

// Given zone step counts and total steps, return divider positions as
// fractional X (0-1 in plot space). Returns [compEnd, midEnd] — two dividers.
// compEnd = fraction where Comp ends / Mid begins
// midEnd  = fraction where Mid ends / Detail begins
function zonesToFractions(zones, n) {
    if (!zones || n < 2) return [0.33, 0.66];
    const total = zones.comp + zones.mid + zones.detail;
    if (total === 0) return [0.33, 0.66];
    return [zones.comp / total, (zones.comp + zones.mid) / total];
}

// ─── Interpolation ────────────────────────────────────────────────────────────
function scaleToSteps(base, targetSteps) {
    if (base.length === targetSteps) return [...base];
    if (targetSteps === 1) return [base[0]];
    const result = [];
    for (let i = 0; i < targetSteps; i++) {
        const t = i / (targetSteps - 1);
        const srcIdx = t * (base.length - 1);
        const lo = Math.floor(srcIdx);
        const hi = Math.min(lo + 1, base.length - 1);
        result.push(base[lo] + (base[hi] - base[lo]) * (srcIdx - lo));
    }
    result[result.length - 1] = 0.0;
    return result;
}

function getSigmas(presetKey, steps, autoScale) {
    const preset = PRESETS[presetKey] || PRESETS.zimage;
    const base = preset.base;
    return (autoScale && base.length !== steps) ? scaleToSteps(base, steps) : [...base];
}

// ─── Graph layout ─────────────────────────────────────────────────────────────
const GRAPH_H   = 170;
const GRAPH_PAD = 6;
const PAD = { top: 28, right: 10, bottom: 26, left: 36 };
// Extra top margin so divider handles sit inside the graph box comfortably
const HANDLE_Y_OFFSET = 8; // handle sits PAD.top/2 from top of graph box

function graphLayout(node, graphY, sigmas) {
    const gx = GRAPH_PAD;
    const gy = graphY;
    const gw = node.size[0] - GRAPH_PAD * 2;
    const gh = GRAPH_H;

    const px = gx + PAD.left;
    const py = gy + PAD.top;
    const pw = gw - PAD.left - PAD.right;
    const ph = gh - PAD.top - PAD.bottom;

    const n = sigmas.length;
    const maxSigma = Math.max(...sigmas, 0.01);

    const dotX      = (i) => px + (n > 1 ? (i / (n - 1)) * pw : pw / 2);
    const dotY      = (s) => py + ph - (s / maxSigma) * ph;
    const sigmaFromY = (y) => Math.max(0, Math.min(maxSigma, ((py + ph - y) / ph) * maxSigma));
    // Convert divider fraction (0-1) to canvas X
    const divX      = (f) => px + f * pw;
    // Handle is a small diamond at the top of the plot area
    const handleY   = gy + HANDLE_Y_OFFSET;

    return { gx, gy, gw, gh, px, py, pw, ph, maxSigma, dotX, dotY, sigmaFromY, divX, handleY };
}

// ─── Draw ─────────────────────────────────────────────────────────────────────
function drawGraph(ctx, node, graphY, sigmas, presetKey, hoveredDot, draggedDot, dividers, hoveredDiv, draggedDiv) {
    const L = graphLayout(node, graphY, sigmas);
    const { gx, gy, gw, gh, px, py, pw, ph, maxSigma, dotX, dotY, divX, handleY } = L;
    const n = sigmas.length;
    // dividers = [compEndFrac, midEndFrac] in 0-1 plot space

    // ── Background ──────────────────────────────────────────────────────────
    ctx.fillStyle = "#0f172a";
    ctx.beginPath(); ctx.roundRect(gx, gy, gw, gh, 4); ctx.fill();
    ctx.strokeStyle = "#1e3a5f"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(gx+0.5, gy+0.5, gw-1, gh-1, 4); ctx.stroke();

    // ── Zone background fills ────────────────────────────────────────────────
    // Three zones: [px, divX(d0)], [divX(d0), divX(d1)], [divX(d1), px+pw]
    const zoneRanges = [
        [px,            divX(dividers[0]), ZONE_COLORS.comp],
        [divX(dividers[0]), divX(dividers[1]), ZONE_COLORS.mid],
        [divX(dividers[1]), px + pw,        ZONE_COLORS.detail],
    ];
    for (const [x0, x1, col] of zoneRanges) {
        if (x1 <= x0) continue;
        ctx.fillStyle = col.fill;
        ctx.fillRect(x0, py, x1 - x0, ph);
    }

    // ── Grid lines + Y labels ────────────────────────────────────────────────
    const gridCount = 4;
    for (let g = 0; g <= gridCount; g++) {
        const lineY = py + (g / gridCount) * ph;
        ctx.strokeStyle = "#1e3a5f"; ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(px, lineY); ctx.lineTo(px + pw, lineY); ctx.stroke();
        ctx.setLineDash([]);
        const val = maxSigma * (1 - g / gridCount);
        ctx.fillStyle = "#475569"; ctx.font = "9px monospace"; ctx.textAlign = "right";
        ctx.fillText(val.toFixed(2), px - 3, lineY + 3);
    }
    ctx.textAlign = "left";

    // ── Zone divider lines ───────────────────────────────────────────────────
    const divNames = ["Comp | Mid", "Mid | Detail"];
    for (let d = 0; d < 2; d++) {
        const x = divX(dividers[d]);
        const isHov = hoveredDiv === d;
        const isDrag = draggedDiv === d;
        const col = d === 0 ? ZONE_COLORS.mid.line : ZONE_COLORS.detail.line;

        ctx.strokeStyle = (isDrag || isHov) ? col : col.replace("0.7", "0.4");
        ctx.lineWidth = isDrag ? 2 : 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x, py + ph); ctx.stroke();
        ctx.setLineDash([]);

        // ── Draggable handle (diamond) at top ─────────────────────────────
        const hx = x, hy = handleY;
        const hs = isDrag ? 7 : isHov ? 6 : 5;
        ctx.beginPath();
        ctx.moveTo(hx,      hy - hs);
        ctx.lineTo(hx + hs, hy);
        ctx.lineTo(hx,      hy + hs);
        ctx.lineTo(hx - hs, hy);
        ctx.closePath();
        ctx.fillStyle = isDrag ? col : isHov ? col.replace("0.7","0.9") : col.replace("0.7","0.55");
        ctx.fill();
        ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1;
        ctx.stroke();
    }

    // ── Zone labels at top ───────────────────────────────────────────────────
    // Comp label
    const compW = divX(dividers[0]) - px;
    if (compW > 20) {
        ctx.fillStyle = ZONE_COLORS.comp.label; ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
        ctx.fillText("COMP", px + compW / 2, gy + PAD.top - 4);
    }
    // Mid label
    const midX0 = divX(dividers[0]), midX1 = divX(dividers[1]);
    if (midX1 - midX0 > 20) {
        ctx.fillStyle = ZONE_COLORS.mid.label; ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
        ctx.fillText("MID", (midX0 + midX1) / 2, gy + PAD.top - 4);
    }
    // Detail label
    const detW = (px + pw) - divX(dividers[1]);
    if (detW > 20) {
        ctx.fillStyle = ZONE_COLORS.detail.label; ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
        ctx.fillText("DETAIL", divX(dividers[1]) + detW / 2, gy + PAD.top - 4);
    }
    ctx.textAlign = "left";

    // ── Filled area under curve ──────────────────────────────────────────────
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

    // ── Curve ────────────────────────────────────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {
        i === 0 ? ctx.moveTo(dotX(i), dotY(sigmas[i])) : ctx.lineTo(dotX(i), dotY(sigmas[i]));
    }
    ctx.stroke();

    // ── Sigma dots ───────────────────────────────────────────────────────────
    const DOT_R = 5;
    for (let i = 0; i < n; i++) {
        const dx = dotX(i), dy = dotY(sigmas[i]);
        const isLast    = i === n - 1;
        const isDragging = draggedDot === i;
        const isHovered  = hoveredDot === i;

        ctx.beginPath();
        ctx.arc(dx, dy, isDragging ? DOT_R+2 : isHovered ? DOT_R+1 : DOT_R, 0, Math.PI*2);
        ctx.fillStyle = isDragging ? "#f87171" : isHovered ? "#93c5fd" : isLast ? "#64748b" : "#3b82f6";
        ctx.fill();
        ctx.strokeStyle = isDragging ? "#fca5a5" : "#0f172a"; ctx.lineWidth = 1.5;
        ctx.stroke();

        if (isHovered || isDragging) {
            const label = sigmas[i].toFixed(3);
            ctx.font = "bold 10px monospace";
            const tw = ctx.measureText(label).width;
            const lx = Math.min(dx - tw/2, px + pw - tw - 2);
            const ly = dy - DOT_R - 6;
            ctx.fillStyle = "rgba(15,23,42,0.9)";
            ctx.fillRect(lx-2, ly-10, tw+4, 13);
            ctx.fillStyle = "#e2e8f0"; ctx.textAlign = "left";
            ctx.fillText(label, lx, ly);
        }
    }
    ctx.textAlign = "left";

    // ── X labels ─────────────────────────────────────────────────────────────
    const labelEvery = Math.max(1, Math.floor(n / 8));
    ctx.fillStyle = "#475569"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    for (let i = 0; i < n; i += labelEvery) ctx.fillText(i, dotX(i), py + ph + 12);
    ctx.textAlign = "left";

    // ── Header bar: preset name + zones ─────────────────────────────────────
    ctx.fillStyle = "#93c5fd"; ctx.font = "bold 10px Arial";
    ctx.fillText(presetKey, gx + 4, gy + 12);

    // Step counts from current divider positions
    const c = Math.round(dividers[0] * (n - 1));
    const m = Math.round((dividers[1] - dividers[0]) * (n - 1));
    const det = (n - 1) - c - m;
    ctx.fillStyle = "#475569"; ctx.font = "9px Arial"; ctx.textAlign = "right";
    ctx.fillText(`C:${c} M:${m} D:${det}`, gx + gw - 4, gy + 12);
    ctx.textAlign = "left";

    // ── Footer ────────────────────────────────────────────────────────────────
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(gx, gy + gh - 14, gw, 14);
    let footerText;
    if (draggedDot >= 0) {
        footerText = `step ${draggedDot}  σ = ${sigmas[draggedDot].toFixed(4)}  (right-click resets dot)`;
        ctx.fillStyle = "#fbbf24";
    } else if (draggedDiv >= 0) {
        const divLabel = draggedDiv === 0 ? "Comp|Mid" : "Mid|Detail";
        footerText = `${divLabel} divider  (drag to resize zones)`;
        ctx.fillStyle = draggedDiv === 0 ? ZONE_COLORS.mid.label : ZONE_COLORS.detail.label;
    } else {
        const info = PRESETS[presetKey] || {};
        footerText = info.description || "";
        ctx.fillStyle = "#64748b";
    }
    ctx.font = "9px Arial";
    ctx.fillText(footerText, gx + 4, gy + gh - 3);
}

// ─── Extension ───────────────────────────────────────────────────────────────
app.registerExtension({
    name: "GraftingRayman.GRSigmaPresetSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GR Sigma Preset Selector") return;

        const DOT_HIT_R = 8;
        const DIV_HIT_R = 8; // horizontal hit radius for divider handles

        // ── helpers ──────────────────────────────────────────────────────────

        nodeType.prototype._graphY = function () {
            let bottom = 0;
            if (this.widgets) {
                for (const w of this.widgets) {
                    const y = (w.last_y ?? 0) + 24;
                    if (y > bottom) bottom = y;
                }
            }
            return bottom + GRAPH_PAD;
        };

        nodeType.prototype._presetSigmas = function () {
            const presetW = this.widgets?.find(w => w.name === "preset");
            const stepsW  = this.widgets?.find(w => w.name === "steps");
            const autoW   = this.widgets?.find(w => w.name === "auto_scale_to_steps");
            return getSigmas(
                presetW?.value  ?? "zimage",
                Number(stepsW?.value ?? 10),
                autoW?.value    ?? true
            );
        };

        nodeType.prototype._presetDividers = function () {
            const presetW = this.widgets?.find(w => w.name === "preset");
            const n = this._sigmas?.length ?? 10;
            const info = PRESETS[presetW?.value] || {};
            const zones = parseZones(info.zones);
            return zonesToFractions(zones, n);
        };

        // Returns widget fingerprint string
        nodeType.prototype._widgetSig = function () {
            const presetW = this.widgets?.find(w => w.name === "preset");
            const stepsW  = this.widgets?.find(w => w.name === "steps");
            const autoW   = this.widgets?.find(w => w.name === "auto_scale_to_steps");
            return `${presetW?.value}|${stepsW?.value}|${autoW?.value}`;
        };

        nodeType.prototype._syncFromWidgets = function () {
            const sig = this._widgetSig();
            if (sig !== this._lastPresetSig) {
                this._lastPresetSig = sig;
                this._sigmas    = this._presetSigmas();
                this._dividers  = this._presetDividers();
                this._hoveredDot = -1;
                this._hoveredDiv = -1;
                return true;
            }
            return false;
        };

        // ── lifecycle ────────────────────────────────────────────────────────

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this._sigmas      = [];
            this._dividers    = [0.33, 0.66]; // fractional X positions [compEnd, midEnd]
            this._draggedDot  = -1;
            this._hoveredDot  = -1;
            this._draggedDiv  = -1;  // 0 = comp|mid divider, 1 = mid|detail divider
            this._hoveredDiv  = -1;
            this._lastPresetSig = null;

            setTimeout(() => {
                this._sigmas   = this._presetSigmas();
                this._dividers = this._presetDividers();
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
            if (!this._sigmas || this._sigmas.length === 0) return;

            const gy = this._graphY();
            const needed = gy + GRAPH_H + GRAPH_PAD;
            if (this.size[1] < needed) this.size[1] = needed;

            const presetKey = this.widgets?.find(w => w.name === "preset")?.value ?? "zimage";
            drawGraph(ctx, this, gy, this._sigmas, presetKey,
                this._hoveredDot, this._draggedDot,
                this._dividers,   this._hoveredDiv, this._draggedDiv);
        };

        // ── mouse ────────────────────────────────────────────────────────────

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos) {
            if (!this._sigmas?.length) return onMouseDown?.apply(this, arguments);

            const gy = this._graphY();
            const L  = graphLayout(this, gy, this._sigmas);

            // ── Check divider handles first (priority over dots) ─────────────
            for (let d = 0; d < 2; d++) {
                const hx = L.divX(this._dividers[d]);
                const hy = L.handleY;
                if (Math.abs(pos[0] - hx) <= DIV_HIT_R && Math.abs(pos[1] - hy) <= DIV_HIT_R) {
                    this._draggedDiv = d;
                    this._hoveredDiv = d;
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            // ── Check sigma dots ─────────────────────────────────────────────
            const n = this._sigmas.length;
            for (let i = 0; i < n; i++) {
                const dx = L.dotX(i) - pos[0];
                const dy = L.dotY(this._sigmas[i]) - pos[1];
                if (Math.hypot(dx, dy) <= DOT_HIT_R) {
                    if (e.button === 2) {
                        const preset = this._presetSigmas();
                        if (i < preset.length) this._sigmas[i] = preset[i];
                        this.setDirtyCanvas(true, true);
                        return true;
                    }
                    this._draggedDot = i;
                    this._hoveredDot = i;
                    this.setDirtyCanvas(true, true);
                    return true;
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

            // ── Dragging a divider ───────────────────────────────────────────
            if (this._draggedDiv >= 0) {
                const d = this._draggedDiv;
                // Convert mouse X to fraction, clamp with 1-step margin from edges and other divider
                const minGap = n > 1 ? 1 / (n - 1) : 0.05;
                let frac = (pos[0] - L.px) / L.pw;
                if (d === 0) {
                    frac = Math.max(minGap, Math.min(this._dividers[1] - minGap, frac));
                } else {
                    frac = Math.max(this._dividers[0] + minGap, Math.min(1 - minGap, frac));
                }
                this._dividers[d] = frac;
                document.body.style.cursor = "ew-resize";
                this.setDirtyCanvas(true, true);
                return true;
            }

            // ── Dragging a sigma dot ─────────────────────────────────────────
            if (this._draggedDot >= 0) {
                const i = this._draggedDot;
                let newSigma = L.sigmaFromY(pos[1]);
                if (i === n - 1) {
                    newSigma = 0;
                } else {
                    if (i > 0)     newSigma = Math.min(newSigma, this._sigmas[i - 1]);
                    if (i < n - 1) newSigma = Math.max(newSigma, this._sigmas[i + 1]);
                }
                this._sigmas[i] = Math.round(newSigma * 100000) / 100000;
                document.body.style.cursor = "ns-resize";
                this.setDirtyCanvas(true, true);
                return true;
            }

            // ── Hover detection ──────────────────────────────────────────────
            let newHovDiv = -1;
            for (let d = 0; d < 2; d++) {
                const hx = L.divX(this._dividers[d]);
                const hy = L.handleY;
                if (Math.abs(pos[0] - hx) <= DIV_HIT_R && Math.abs(pos[1] - hy) <= DIV_HIT_R) {
                    newHovDiv = d; break;
                }
            }

            let newHovDot = -1;
            if (newHovDiv < 0) {
                for (let i = 0; i < n; i++) {
                    if (Math.hypot(L.dotX(i) - pos[0], L.dotY(this._sigmas[i]) - pos[1]) <= DOT_HIT_R) {
                        newHovDot = i; break;
                    }
                }
            }

            const changed = newHovDiv !== this._hoveredDiv || newHovDot !== this._hoveredDot;
            if (changed) {
                this._hoveredDiv = newHovDiv;
                this._hoveredDot = newHovDot;
                document.body.style.cursor = newHovDiv >= 0 ? "ew-resize" : newHovDot >= 0 ? "ns-resize" : "";
                this.setDirtyCanvas(true, true);
            }

            return onMouseMove?.apply(this, arguments);
        };

        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e) {
            if (this._draggedDiv >= 0 || this._draggedDot >= 0) {
                this._draggedDiv = -1;
                this._draggedDot = -1;
                document.body.style.cursor = "";
                this.setDirtyCanvas(true, true);
                return true;
            }
            return onMouseUp?.apply(this, arguments);
        };

        const onMouseLeave = nodeType.prototype.onMouseLeave;
        nodeType.prototype.onMouseLeave = function () {
            this._hoveredDot = -1;
            this._hoveredDiv = -1;
            this._draggedDot = -1;
            this._draggedDiv = -1;
            document.body.style.cursor = "";
            this.setDirtyCanvas(true, true);
            return onMouseLeave?.apply(this, arguments);
        };

        // ── size ─────────────────────────────────────────────────────────────

        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = computeSize ? computeSize.apply(this, arguments) : [240, 120];
            const needed = (this._graphY?.() ?? size[1]) + GRAPH_H + GRAPH_PAD;
            if (size[1] < needed) size[1] = needed;
            return size;
        };
    }
});