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
            break;
        case "log":
            y = x.map(v => Math.log1p(v * exponent));
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
    const yMax = Math.max(...y, 1e-10);
    if (curveType === "cosine" || curveType === "poly") {
        // these don't normalise in Python either
    } else {
        y = y.map(v => v / yMax);
    }
    return y.map(v => zoneMax - (zoneMax - zoneMin) * v);
}

function enforceMonotonic(sigmas) {
    for (let i = 1; i < sigmas.length; i++) {
        if (sigmas[i] >= sigmas[i - 1]) sigmas[i] = sigmas[i - 1] - 1e-6;
    }
    return sigmas;
}

function generateSigmasFromPreset(presetDef, overallMax, overallMin) {
    const { comp_thresh, mid_thresh,
            comp_steps,  comp_curve,  comp_exponent,
            mid_steps,   mid_curve,   mid_exponent,
            detail_steps,detail_curve,detail_exponent } = presetDef;

    const compMin  = Math.max(overallMax * comp_thresh, mid_thresh  * 1.1);
    const midMin   = Math.max(overallMax * mid_thresh,  overallMin  * 1.1);

    const compSig   = makeSegment(comp_steps,   comp_curve,   overallMax, compMin, comp_exponent);
    const midSig    = makeSegment(mid_steps,    mid_curve,    compMin,    midMin,  mid_exponent);
    const detailSig = makeSegment(detail_steps, detail_curve, midMin,     overallMin, detail_exponent);

    return enforceMonotonic([...compSig, ...midSig, ...detailSig]);
}

// Z-Image exact sigmas (from GRSigmas.py generate_zimage_sigmas)
function generateZImageSigmas(totalSteps) {
    let sigmas;
    if (totalSteps >= 9)     sigmas = [0.991,0.98,0.92,0.935,0.90,0.875,0.750,0.6582,0.4556,0.2000,0.0000];
    else if (totalSteps===8) sigmas = [0.991,0.98,0.92,0.935,0.90,0.875,0.750,0.6582,0.3019,0.0000];
    else if (totalSteps===7) sigmas = [0.991,0.98,0.92,0.9350,0.8916,0.7600,0.6582,0.3019,0.0000];
    else if (totalSteps===6) sigmas = [0.991,0.980,0.920,0.942,0.780,0.6582,0.3019,0.0000];
    else if (totalSteps===5) sigmas = [0.991,0.980,0.920,0.942,0.780,0.6200,0.0000];
    else                     sigmas = [0.991,0.980,0.920,0.942,0.790,0.0000];
    while (sigmas.length > totalSteps + 1) sigmas.pop();
    return sigmas;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRESET DEFINITIONS — exact copy of GRSigmas.py apply_preset()
// ═══════════════════════════════════════════════════════════════════════════════
const PRESET_DEFS = {
    balanced:          { comp_thresh:0.75, mid_thresh:0.45, comp_steps:8,  comp_curve:"exp",    comp_exponent:2.0, mid_steps:10, mid_curve:"linear", mid_exponent:1.0, detail_steps:6,  detail_curve:"log",    detail_exponent:1.0 },
    composition_heavy: { comp_thresh:0.85, mid_thresh:0.40, comp_steps:12, comp_curve:"exp",    comp_exponent:2.5, mid_steps:8,  mid_curve:"cosine",  mid_exponent:1.5, detail_steps:4,  detail_curve:"log",    detail_exponent:0.8 },
    detail_heavy:      { comp_thresh:0.65, mid_thresh:0.35, comp_steps:4,  comp_curve:"exp",    comp_exponent:1.5, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:12, detail_curve:"log",    detail_exponent:0.5 },
    aggressive:        { comp_thresh:0.90, mid_thresh:0.60, comp_steps:10, comp_curve:"poly",   comp_exponent:3.0, mid_steps:8,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:6,  detail_curve:"log",    detail_exponent:1.5 },
    subtle:            { comp_thresh:0.70, mid_thresh:0.40, comp_steps:6,  comp_curve:"cosine", comp_exponent:1.0, mid_steps:12, mid_curve:"linear",  mid_exponent:1.0, detail_steps:6,  detail_curve:"log",    detail_exponent:0.8 },
    portrait:          { comp_thresh:0.80, mid_thresh:0.40, comp_steps:10, comp_curve:"cosine", comp_exponent:1.5, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log",    detail_exponent:0.7 },
    landscape:         { comp_thresh:0.70, mid_thresh:0.35, comp_steps:6,  comp_curve:"exp",    comp_exponent:1.8, mid_steps:12, mid_curve:"cosine",  mid_exponent:1.2, detail_steps:6,  detail_curve:"log",    detail_exponent:0.5 },
    architecture:      { comp_thresh:0.85, mid_thresh:0.50, comp_steps:12, comp_curve:"poly",   comp_exponent:2.5, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log",    detail_exponent:1.5 },
    abstract:          { comp_thresh:0.65, mid_thresh:0.30, comp_steps:5,  comp_curve:"exp",    comp_exponent:1.2, mid_steps:8,  mid_curve:"cosine",  mid_exponent:0.8, detail_steps:11, detail_curve:"log",    detail_exponent:0.3 },
    fine_detail:       { comp_thresh:0.60, mid_thresh:0.25, comp_steps:4,  comp_curve:"exp",    comp_exponent:1.0, mid_steps:8,  mid_curve:"linear",  mid_exponent:1.0, detail_steps:12, detail_curve:"log",    detail_exponent:0.2 },
    fast_decay:        { comp_thresh:0.90, mid_thresh:0.60, comp_steps:4,  comp_curve:"poly",   comp_exponent:3.0, mid_steps:6,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:14, detail_curve:"log",    detail_exponent:1.0 },
    slow_decay:        { comp_thresh:0.70, mid_thresh:0.40, comp_steps:10, comp_curve:"cosine", comp_exponent:1.0, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log",    detail_exponent:0.5 },
    mid_centric:       { comp_thresh:0.75, mid_thresh:0.35, comp_steps:6,  comp_curve:"exp",    comp_exponent:1.5, mid_steps:14, mid_curve:"linear",  mid_exponent:1.0, detail_steps:4,  detail_curve:"log",    detail_exponent:0.8 },
    high_contrast:     { comp_thresh:0.85, mid_thresh:0.50, comp_steps:8,  comp_curve:"poly",   comp_exponent:3.0, mid_steps:8,  mid_curve:"exp",     mid_exponent:2.0, detail_steps:8,  detail_curve:"log",    detail_exponent:1.5 },
    low_contrast:      { comp_thresh:0.70, mid_thresh:0.40, comp_steps:8,  comp_curve:"cosine", comp_exponent:1.0, mid_steps:10, mid_curve:"linear",  mid_exponent:1.0, detail_steps:6,  detail_curve:"log",    detail_exponent:0.5 },
};

// GRSigmaPresets manual presets (from GRSigmas.py GRSigmaPresets class)
const MANUAL_PRESETS = {
    ultra_lock:      [0.0],
    micro_detail:    [0.1, 0.0],
    low_motion:      [0.2, 0.0],
    img2img_safe:    [0.3, 0.0],
    balanced_i2v:    [0.5, 0.25, 0.0],
    stylised_motion: [0.7, 0.4, 0.2, 0.0],
    manual_preset:   [0.909375, 0.725, 0.421875, 0.0],
    mid_sigma_focus: [1.0, 0.6, 0.2, 0.0],
    high_detail_tail:[0.6, 0.3, 0.1, 0.0],
    experimental_wide:[1.2, 0.8, 0.4, 0.0],
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

// Generate sigmas matching GRSigmas.py exactly
// overall_max/min default to 1.0/0.01 matching GRSigmas defaults
function getSigmas(presetKey, overallMax = 1.0, overallMin = 0.01) {
    if (presetKey === "zimage") return generateZImageSigmas(9);
    if (PRESET_DEFS[presetKey]) return generateSigmasFromPreset(PRESET_DEFS[presetKey], overallMax, overallMin);
    if (MANUAL_PRESETS[presetKey]) return [...MANUAL_PRESETS[presetKey]];
    return generateSigmasFromPreset(PRESET_DEFS.balanced, overallMax, overallMin);
}

function zonesToFractions(presetKey, n) {
    const z = PRESET_ZONE_STEPS[presetKey];
    if (!z || n < 2) return [0.33, 0.66];
    const total = z.comp + z.mid + z.detail;
    return [z.comp / total, (z.comp + z.mid) / total];
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZONE COLORS / CURVE SHAPES
// ═══════════════════════════════════════════════════════════════════════════════
const ZONE_COLORS = {
    comp:   { fill:"rgba(239,68,68,0.07)",  line:"rgba(239,68,68,0.75)",  label:"#f87171" },
    mid:    { fill:"rgba(234,179,8,0.07)",  line:"rgba(234,179,8,0.75)",  label:"#fbbf24" },
    detail: { fill:"rgba(34,197,94,0.07)",  line:"rgba(34,197,94,0.75)",  label:"#4ade80" },
};

const CURVE_SHAPES = [
    { id:"preset",  label:"Pre", fn: null },  // use preset's own curve — no reshaping
    { id:"linear",  label:"Lin", fn: t => t },
    { id:"easein",  label:"In",  fn: t => t*t },
    { id:"easeout", label:"Out", fn: t => 1-(1-t)*(1-t) },
    { id:"cosine",  label:"Cos", fn: t => (1-Math.cos(t*Math.PI))/2 },
    { id:"s-curve", label:"S",   fn: t => t<0.5 ? 2*t*t : 1-2*(1-t)*(1-t) },
];

// Remap rawSigmas through shapeFn (t→[0,1]), preserving start/end.
// "preset" shape (fn=null) returns the raw sigmas untouched.
function deriveDisplaySigmas(rawSigmas, activeShape) {
    const shape = CURVE_SHAPES.find(s => s.id === activeShape);
    if (!shape || !shape.fn) return [...rawSigmas];
    const n = rawSigmas.length;
    if (n < 3) return [...rawSigmas];
    const start = rawSigmas[0], end = rawSigmas[n - 1];
    return rawSigmas.map((_, i) => {
        if (i === n - 1) return end;
        return start + (end - start) * shape.fn(i / (n - 1));
    });
}

function snapToStep(frac, n) {
    if (n < 2) return frac;
    return Math.round(frac * (n-1)) / (n-1);
}

function dotZone(i, n, dividers) {
    if (n < 2) return "comp";
    const f = i / (n-1);
    return f < dividers[0] ? "comp" : f < dividers[1] ? "mid" : "detail";
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYOUT
// ═══════════════════════════════════════════════════════════════════════════════
const GRAPH_H   = 190;
const GRAPH_PAD = 6;
const PAD       = { top:42, right:10, bottom:28, left:36 };

function graphLayout(node, graphY, sigmas) {
    const gx = GRAPH_PAD;
    const gy = graphY;
    const gw = Math.max(node.size[0] - GRAPH_PAD*2, 80);
    const gh = GRAPH_H;
    const px = gx + PAD.left, py = gy + PAD.top;
    const pw = gw - PAD.left - PAD.right;
    const ph = gh - PAD.top  - PAD.bottom;
    const n  = sigmas.length;
    const maxSigma = Math.max(...sigmas, 0.01);

    const dotX       = i  => px + (n>1 ? (i/(n-1))*pw : pw/2);
    const dotY       = s  => py + ph - (s/maxSigma)*ph;
    const sigmaFromY = y  => Math.max(0, Math.min(maxSigma, ((py+ph-y)/ph)*maxSigma));
    const divX       = f  => px + f*pw;
    const handleY    = gy + PAD.top - 8;
    const tbY        = gy + 14;
    const tbH        = 14;
    const resetBtn   = { x:gx+gw-22, y:gy+2, w:20, h:12 };
    const copyBtn    = { x:gx+gw-44, y:gy+2, w:20, h:12 };
    const mirrorBtn  = { x:gx+gw-68, y:gy+2, w:22, h:12 };
    const shapeBtns  = CURVE_SHAPES.map((s,i)=>({ ...s, x:gx+4+i*22, y:tbY, w:20, h:tbH }));

    return { gx,gy,gw,gh,px,py,pw,ph, maxSigma,dotX,dotY,sigmaFromY,divX,handleY,
             tbY,tbH, resetBtn,copyBtn,mirrorBtn,shapeBtns };
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAW
// ═══════════════════════════════════════════════════════════════════════════════
function drawGraph(ctx, node, graphY, sigmas, presetKey,
                   hoveredDot, draggedDot, dividers, hoveredDiv, draggedDiv,
                   mirrorMode, activeShape, copyFlash) {

    const L = graphLayout(node, graphY, sigmas);
    const { gx,gy,gw,gh,px,py,pw,ph,maxSigma,dotX,dotY,divX,handleY,
            resetBtn,copyBtn,mirrorBtn,shapeBtns } = L;
    const n = sigmas.length;

    // Background + border
    ctx.fillStyle="#0f172a"; ctx.fillRect(gx,gy,gw,gh);
    ctx.strokeStyle="#1e3a5f"; ctx.lineWidth=1; ctx.strokeRect(gx+.5,gy+.5,gw-1,gh-1);

    // Zone fills
    for (const [x0,x1,col] of [
        [px,            divX(dividers[0]), ZONE_COLORS.comp],
        [divX(dividers[0]),divX(dividers[1]),ZONE_COLORS.mid],
        [divX(dividers[1]),px+pw,          ZONE_COLORS.detail],
    ]) {
        if (x1>x0) { ctx.fillStyle=col.fill; ctx.fillRect(x0,py,x1-x0,ph); }
    }

    // Grid
    for (let g=0; g<=4; g++) {
        const ly = py+(g/4)*ph;
        ctx.strokeStyle="#1e3a5f"; ctx.lineWidth=1;
        ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(px,ly); ctx.lineTo(px+pw,ly); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle="#475569"; ctx.font="9px monospace"; ctx.textAlign="right";
        ctx.fillText((maxSigma*(1-g/4)).toFixed(3), px-3, ly+3);
    }
    ctx.textAlign="left";

    // Divider lines + handles
    for (let d=0; d<2; d++) {
        const x=divX(dividers[d]), col=d===0?ZONE_COLORS.mid.line:ZONE_COLORS.detail.line;
        const hot=draggedDiv===d||hoveredDiv===d;
        ctx.strokeStyle=hot?col:col.replace("0.75","0.35"); ctx.lineWidth=hot?2:1.5;
        ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(x,py); ctx.lineTo(x,py+ph); ctx.stroke(); ctx.setLineDash([]);
        const hs=draggedDiv===d?7:hoveredDiv===d?6:5, hy=handleY;
        ctx.beginPath(); ctx.moveTo(x,hy-hs); ctx.lineTo(x+hs,hy); ctx.lineTo(x,hy+hs); ctx.lineTo(x-hs,hy); ctx.closePath();
        ctx.fillStyle=hot?col:col.replace("0.75","0.5"); ctx.fill();
        ctx.strokeStyle="#0f172a"; ctx.lineWidth=1; ctx.stroke();
    }

    // Zone labels
    const labelY=py-4;
    for (const [x0,x1,col,lbl] of [
        [px,               divX(dividers[0]), ZONE_COLORS.comp.label,   "COMP"],
        [divX(dividers[0]),divX(dividers[1]), ZONE_COLORS.mid.label,    "MID"],
        [divX(dividers[1]),px+pw,             ZONE_COLORS.detail.label, "DETAIL"],
    ]) {
        if (x1-x0<18) continue;
        ctx.fillStyle=col; ctx.font="bold 8px Arial"; ctx.textAlign="center";
        ctx.fillText(lbl,(x0+x1)/2,labelY);
    }
    ctx.textAlign="left";

    // Curve fill
    if (n>1) {
        ctx.beginPath(); ctx.moveTo(dotX(0),py+ph);
        for (let i=0;i<n;i++) ctx.lineTo(dotX(i),dotY(sigmas[i]));
        ctx.lineTo(dotX(n-1),py+ph); ctx.closePath();
        const g=ctx.createLinearGradient(0,py,0,py+ph);
        g.addColorStop(0,"rgba(59,130,246,0.35)"); g.addColorStop(1,"rgba(59,130,246,0.03)");
        ctx.fillStyle=g; ctx.fill();
    }

    // Curve line
    ctx.beginPath(); ctx.strokeStyle="#3b82f6"; ctx.lineWidth=2;
    for (let i=0;i<n;i++) i===0?ctx.moveTo(dotX(i),dotY(sigmas[i])):ctx.lineTo(dotX(i),dotY(sigmas[i]));
    ctx.stroke();

    // Mirror ghost
    if (mirrorMode && draggedDot>=0) {
        ctx.beginPath(); ctx.strokeStyle="rgba(251,191,36,0.25)"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
        for (let i=0;i<n;i++) { const mi=n-1-i; i===0?ctx.moveTo(dotX(i),dotY(sigmas[mi])):ctx.lineTo(dotX(i),dotY(sigmas[mi])); }
        ctx.stroke(); ctx.setLineDash([]);
    }

    // Dots
    for (let i=0;i<n;i++) {
        const dx=dotX(i), dy=dotY(sigmas[i]);
        const isDrag=draggedDot===i, isHov=hoveredDot===i, isLast=i===n-1;
        const zone=dotZone(i,n,dividers);
        const isMirror=mirrorMode&&draggedDot>=0&&i===(n-1-draggedDot)&&i!==draggedDot;
        ctx.beginPath(); ctx.arc(dx,dy,isDrag?7:isHov?6:5,0,Math.PI*2);
        ctx.fillStyle=isDrag?"#f87171":isMirror?"rgba(251,191,36,0.7)":isHov?"#93c5fd":isLast?"#334155":"#3b82f6";
        ctx.fill(); ctx.strokeStyle=isDrag?"#fca5a5":isHov?"#bfdbfe":"#0f172a"; ctx.lineWidth=1.5; ctx.stroke();

        if (isHov||isDrag) {
            const zn=zone.toUpperCase();
            const d0s=Math.round(dividers[0]*(n-1)), d1s=Math.round(dividers[1]*(n-1));
            const stepStr=i<d0s?`${zn} ${i+1}/${d0s}`:i<d1s?`${zn} ${i-d0s+1}/${d1s-d0s}`:`${zn} ${i-d1s+1}/${n-1-d1s}`;
            const lbl=`${stepStr}  σ=${sigmas[i].toFixed(4)}`;
            ctx.font="bold 9px monospace";
            const tw=ctx.measureText(lbl).width;
            let lx=Math.max(px,Math.min(px+pw-tw-2,dx-tw/2));
            ctx.fillStyle="rgba(15,23,42,0.92)"; ctx.fillRect(lx-3,dy-18,tw+6,14);
            ctx.fillStyle=ZONE_COLORS[zone]?.label??"#e2e8f0"; ctx.textAlign="left"; ctx.fillText(lbl,lx,dy-7);
        }
    }
    ctx.textAlign="left";

    // X labels
    const every=Math.max(1,Math.floor(n/8));
    ctx.fillStyle="#475569"; ctx.font="9px monospace"; ctx.textAlign="center";
    for (let i=0;i<n;i+=every) ctx.fillText(i,dotX(i),py+ph+12);
    ctx.textAlign="left";

    // Header
    ctx.fillStyle="#93c5fd"; ctx.font="bold 10px Arial"; ctx.fillText(presetKey,gx+4,gy+12);
    const c=Math.round(dividers[0]*(n-1)), m=Math.round((dividers[1]-dividers[0])*(n-1)), det=(n-1)-c-m;
    ctx.fillStyle="#475569"; ctx.font="9px Arial"; ctx.textAlign="right";
    ctx.fillText(`C:${c} M:${m} D:${det}`,gx+gw-70,gy+12); ctx.textAlign="left";

    // Shape buttons
    for (const btn of shapeBtns) {
        const on=activeShape===btn.id;
        ctx.fillStyle=on?"#1d4ed8":"#1e293b"; ctx.fillRect(btn.x,btn.y,btn.w,btn.h);
        ctx.strokeStyle=on?"#60a5fa":"#334155"; ctx.lineWidth=1; ctx.strokeRect(btn.x+.5,btn.y+.5,btn.w-1,btn.h-1);
        ctx.fillStyle=on?"#e2e8f0":"#64748b"; ctx.font=on?"bold 8px Arial":"8px Arial"; ctx.textAlign="center";
        ctx.fillText(btn.label,btn.x+btn.w/2,btn.y+btn.h-3);
    }
    ctx.textAlign="left";

    // Mirror button
    const mb=mirrorBtn;
    ctx.fillStyle=mirrorMode?"#78350f":"#1e293b"; ctx.fillRect(mb.x,mb.y,mb.w,mb.h);
    ctx.strokeStyle=mirrorMode?"#fbbf24":"#334155"; ctx.lineWidth=1; ctx.strokeRect(mb.x+.5,mb.y+.5,mb.w-1,mb.h-1);
    ctx.fillStyle=mirrorMode?"#fbbf24":"#64748b"; ctx.font=mirrorMode?"bold 8px Arial":"8px Arial"; ctx.textAlign="center";
    ctx.fillText("⇌ Mir",mb.x+mb.w/2,mb.y+mb.h-3); ctx.textAlign="left";

    // Copy button
    const cb=copyBtn, cf=copyFlash>0;
    ctx.fillStyle=cf?"#14532d":"#1e293b"; ctx.fillRect(cb.x,cb.y,cb.w,cb.h);
    ctx.strokeStyle=cf?"#4ade80":"#334155"; ctx.lineWidth=1; ctx.strokeRect(cb.x+.5,cb.y+.5,cb.w-1,cb.h-1);
    ctx.fillStyle=cf?"#4ade80":"#64748b"; ctx.font="8px Arial"; ctx.textAlign="center";
    ctx.fillText(cf?"✓":"⎘ Copy",cb.x+cb.w/2,cb.y+cb.h-3); ctx.textAlign="left";

    // Reset button
    const rb=resetBtn;
    ctx.fillStyle="#1e293b"; ctx.fillRect(rb.x,rb.y,rb.w,rb.h);
    ctx.strokeStyle="#334155"; ctx.lineWidth=1; ctx.strokeRect(rb.x+.5,rb.y+.5,rb.w-1,rb.h-1);
    ctx.fillStyle="#ef4444"; ctx.font="8px Arial"; ctx.textAlign="center";
    ctx.fillText("↺ All",rb.x+rb.w/2,rb.y+rb.h-3); ctx.textAlign="left";

    // Footer
    ctx.fillStyle="#1e293b"; ctx.fillRect(gx,gy+gh-14,gw,14);
    let ft, fc;
    if (draggedDot>=0)  { ft=`σ[${draggedDot}]=${sigmas[draggedDot].toFixed(5)}${mirrorMode?" [mirror]":""}  right-click=reset`; fc="#fbbf24"; }
    else if (draggedDiv>=0) { ft=`${draggedDiv===0?"Comp|Mid":"Mid|Detail"} divider — snaps to steps`; fc=draggedDiv===0?ZONE_COLORS.mid.label:ZONE_COLORS.detail.label; }
    else { ft=PRESET_DESCRIPTIONS[presetKey]||""; fc="#64748b"; }
    ctx.fillStyle=fc; ctx.font="9px Arial"; ctx.fillText(ft,gx+4,gy+gh-3);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIT TEST
// ═══════════════════════════════════════════════════════════════════════════════
const DOT_HIT=8, DIV_HIT=8;
function hitBtn(pos,L) {
    for (const b of L.shapeBtns) if (pos[0]>=b.x&&pos[0]<=b.x+b.w&&pos[1]>=b.y&&pos[1]<=b.y+b.h) return {type:"shape",id:b.id};
    const {resetBtn:rb,copyBtn:cb,mirrorBtn:mb}=L;
    if (pos[0]>=rb.x&&pos[0]<=rb.x+rb.w&&pos[1]>=rb.y&&pos[1]<=rb.y+rb.h) return {type:"reset"};
    if (pos[0]>=cb.x&&pos[0]<=cb.x+cb.w&&pos[1]>=cb.y&&pos[1]<=cb.y+cb.h) return {type:"copy"};
    if (pos[0]>=mb.x&&pos[0]<=mb.x+mb.w&&pos[1]>=mb.y&&pos[1]<=mb.y+mb.h) return {type:"mirror"};
    return null;
}
function hitDiv(pos,L,divs) {
    for (let d=0;d<2;d++) if (Math.abs(pos[0]-L.divX(divs[d]))<=DIV_HIT&&Math.abs(pos[1]-L.handleY)<=DIV_HIT) return d;
    return -1;
}
function hitDot(pos,L,sigs) {
    for (let i=0;i<sigs.length;i++) if (Math.hypot(L.dotX(i)-pos[0],L.dotY(sigs[i])-pos[1])<=DOT_HIT) return i;
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTENSION
// ═══════════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "GraftingRayman.GRSigmaPresetSelectorAdvanced",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GR Sigma Preset Selector Advanced") return;

        nodeType.prototype._graphY = function () {
            let b=0; if (this.widgets) for (const w of this.widgets) { const y=(w.last_y??0)+24; if(y>b)b=y; } return b+GRAPH_PAD;
        };
        nodeType.prototype._widgetSig = function () {
            const p=this.widgets?.find(w=>w.name==="preset");
            return `${p?.value}`;
        };
        // Write _activeShape back to the hidden curve_shape widget so Python receives it
        nodeType.prototype._syncShapeWidget = function () {
            const w = this.widgets?.find(w => w.name === "curve_shape");
            if (w) w.value = this._activeShape;
        };
        nodeType.prototype._freshSigmas = function () {
            const p=this.widgets?.find(w=>w.name==="preset");
            return getSigmas(p?.value??"balanced");
        };
        nodeType.prototype._freshDividers = function () {
            const p=this.widgets?.find(w=>w.name==="preset");
            return zonesToFractions(p?.value??"balanced", this._sigmas?.length??24);
        };
        nodeType.prototype._syncFromWidgets = function () {
            const sig = this._widgetSig();
            if (sig !== this._wSig) {
                this._wSig = sig;
                this._rawSigmas = this._freshSigmas();
                this._sigmas    = [...this._rawSigmas];
                this._dividers  = this._freshDividers();
                this._undoStack = [];
                this._activeShape = "preset";
                this._syncShapeWidget?.();
                this._hoveredDot = this._hoveredDiv = -1;
            }
        };
        nodeType.prototype._pushUndo = function () {
            this._undoStack=this._undoStack||[];
            this._undoStack.push({ sigmas:[...this._sigmas], raw:[...(this._rawSigmas??this._sigmas)], shape:this._activeShape });
            if (this._undoStack.length>20) this._undoStack.shift();
        };
        nodeType.prototype._undo = function () {
            if (!this._undoStack?.length) return;
            const snap=this._undoStack.pop();
            this._sigmas=snap.sigmas; this._rawSigmas=snap.raw; this._activeShape=snap.shape;
            this.setDirtyCanvas(true,true);
        };

        const onNodeCreated=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){
            const r=onNodeCreated?.apply(this,arguments);
            this._rawSigmas=[]; this._sigmas=[]; this._dividers=[0.33,0.66]; this._undoStack=[];
            this._draggedDot=-1; this._hoveredDot=-1; this._draggedDiv=-1; this._hoveredDiv=-1;
            this._mirrorMode=false; this._activeShape="preset"; this._copyFlash=0; this._wSig=null;
            setTimeout(()=>{ this._rawSigmas=this._freshSigmas(); this._sigmas=[...this._rawSigmas]; this._dividers=this._freshDividers(); this._wSig=this._widgetSig(); this.setDirtyCanvas(true,true); },60);
            return r;
        };

        const onKeyDown=nodeType.prototype.onKeyDown;
        nodeType.prototype.onKeyDown=function(e){
            if ((e.ctrlKey||e.metaKey)&&e.key==="z"){ e.preventDefault(); this._undo(); return true; }
            return onKeyDown?.apply(this,arguments);
        };

        const onDrawForeground=nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground=function(ctx){
            onDrawForeground?.apply(this,arguments);
            this._syncFromWidgets();
            if (!this._sigmas?.length) return;
            const gy=this._graphY();
            if (this.size[1]<gy+GRAPH_H+GRAPH_PAD) this.size[1]=gy+GRAPH_H+GRAPH_PAD;
            if (this._copyFlash>0){ this._copyFlash--; this.setDirtyCanvas(true,true); }
            const pk=this.widgets?.find(w=>w.name==="preset")?.value??"balanced";
            // Derive display sigmas: raw preset curve reshaped by active shape button
            const displaySigmas = deriveDisplaySigmas(this._rawSigmas ?? this._sigmas, this._activeShape);
            // Blend in any manual dot edits stored in _sigmas
            const drawSigmas = this._sigmas.map((s, i) => {
                const raw = this._rawSigmas?.[i] ?? s;
                return (s !== raw) ? s : displaySigmas[i] ?? s;
            });
            drawGraph(ctx,this,gy,drawSigmas,pk,this._hoveredDot,this._draggedDot,
                this._dividers,this._hoveredDiv,this._draggedDiv,
                this._mirrorMode,this._activeShape,this._copyFlash);
        };

        const onMouseDown=nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown=function(e,pos){
            if (!this._sigmas?.length) return onMouseDown?.apply(this,arguments);
            const gy=this._graphY(), L=graphLayout(this,gy,this._sigmas);
            const btn=hitBtn(pos,L);
            if (btn){
                if (btn.type==="shape"){ this._activeShape=btn.id; this._syncShapeWidget(); }
                if (btn.type==="reset"){ this._pushUndo(); this._rawSigmas=this._freshSigmas(); this._sigmas=[...this._rawSigmas]; this._activeShape="preset"; this._syncShapeWidget(); this._dividers=this._freshDividers(); }
                if (btn.type==="copy"){ const cs=deriveDisplaySigmas(this._rawSigmas??this._sigmas,this._activeShape).map((s,i)=>{ const raw=(this._rawSigmas??this._sigmas)[i]??s; return (this._sigmas[i]!==raw)?this._sigmas[i]:s; }); navigator.clipboard?.writeText("["+cs.map(v=>v.toFixed(5)).join(", ")+"]").catch(()=>{}); this._copyFlash=40; }
                if (btn.type==="mirror") this._mirrorMode=!this._mirrorMode;
                this.setDirtyCanvas(true,true); return true;
            }
            const dh=hitDiv(pos,L,this._dividers);
            if (dh>=0){ this._draggedDiv=dh; this._hoveredDiv=dh; this.setDirtyCanvas(true,true); return true; }
            const di=hitDot(pos,L,this._sigmas);
            if (di>=0){
                if (e.button===2||e.detail===2){ this._pushUndo(); if(di<(this._rawSigmas?.length??0))this._sigmas[di]=this._rawSigmas[di]; }
                else { this._pushUndo(); this._draggedDot=di; this._hoveredDot=di; }
                this.setDirtyCanvas(true,true); return true;
            }
            return onMouseDown?.apply(this,arguments);
        };

        const onMouseMove=nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove=function(e,pos){
            if (!this._sigmas?.length) return onMouseMove?.apply(this,arguments);
            const gy=this._graphY(), L=graphLayout(this,gy,this._sigmas), n=this._sigmas.length;
            if (this._draggedDiv>=0){
                const d=this._draggedDiv, gap=n>1?1/(n-1):0.05;
                let f=snapToStep((pos[0]-L.px)/L.pw,n);
                f=d===0?Math.max(gap,Math.min(this._dividers[1]-gap,f)):Math.max(this._dividers[0]+gap,Math.min(1-gap,f));
                this._dividers[d]=f; document.body.style.cursor="ew-resize"; this.setDirtyCanvas(true,true); return true;
            }
            if (this._draggedDot>=0){
                const i=this._draggedDot; let s=L.sigmaFromY(pos[1]);
                if (i===n-1) s=0; else { if(i>0)s=Math.min(s,this._sigmas[i-1]); if(i<n-1)s=Math.max(s,this._sigmas[i+1]); }
                this._sigmas[i]=Math.round(s*100000)/100000;
                if (this._mirrorMode){ const mi=n-1-i; if(mi!==i&&mi>=0&&mi<n-1){ let ms=Math.max(0,Math.min(this._sigmas[0],this._sigmas[0]-s)); if(mi>0)ms=Math.min(ms,this._sigmas[mi-1]); if(mi<n-1)ms=Math.max(ms,this._sigmas[mi+1]); this._sigmas[mi]=Math.round(ms*100000)/100000; } }
                document.body.style.cursor="ns-resize"; this.setDirtyCanvas(true,true); return true;
            }
            const nhd=hitDiv(pos,L,this._dividers), nht=nhd<0?hitDot(pos,L,this._sigmas):-1;
            if (nhd!==this._hoveredDiv||nht!==this._hoveredDot){
                this._hoveredDiv=nhd; this._hoveredDot=nht;
                document.body.style.cursor=nhd>=0?"ew-resize":nht>=0?"ns-resize":"";
                this.setDirtyCanvas(true,true);
            }
            return onMouseMove?.apply(this,arguments);
        };

        const onMouseUp=nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp=function(e){
            if (this._draggedDiv>=0||this._draggedDot>=0){ this._draggedDiv=this._draggedDot=-1; document.body.style.cursor=""; this.setDirtyCanvas(true,true); return true; }
            return onMouseUp?.apply(this,arguments);
        };

        const onMouseLeave=nodeType.prototype.onMouseLeave;
        nodeType.prototype.onMouseLeave=function(){
            this._hoveredDot=this._hoveredDiv=this._draggedDot=this._draggedDiv=-1;
            document.body.style.cursor=""; this.setDirtyCanvas(true,true);
            return onMouseLeave?.apply(this,arguments);
        };

        const computeSize=nodeType.prototype.computeSize;
        nodeType.prototype.computeSize=function(out){
            const s=computeSize?.apply(this,arguments)??[240,120];
            const need=(this._graphY?.()??s[1])+GRAPH_H+GRAPH_PAD;
            if(s[1]<need)s[1]=need; return s;
        };
    }
});