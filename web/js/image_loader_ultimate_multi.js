/**
 * ComfyUI Custom Node - Image Loader Ultimate Multi
 * Place in: ComfyUI/custom_nodes/image_loader_ultimate_multi/image_loader_ultimate_multi.js
 */

import { app } from "/scripts/app.js";

// ── Pure helpers ──────────────────────────────────────────────────────────────

function getRotatedCanvas(img, rotation, flipH, flipV) {
    if (!img) return null;
    const sw = rotation === 90 || rotation === 270;
    const off = document.createElement("canvas");
    off.width  = sw ? (img.naturalHeight||img.height) : (img.naturalWidth||img.width);
    off.height = sw ? (img.naturalWidth||img.width)   : (img.naturalHeight||img.height);
    const oc = off.getContext("2d");
    oc.translate(off.width/2, off.height/2);
    oc.rotate(rotation * Math.PI / 180);
    oc.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    oc.drawImage(img, -(img.naturalWidth||img.width)/2, -(img.naturalHeight||img.height)/2);
    return off;
}

function rotatedDims(img, rotation) {
    if (!img) return { w:0, h:0 };
    const sw = rotation === 90 || rotation === 270;
    return {
        w: sw ? (img.naturalHeight||img.height) : (img.naturalWidth||img.width),
        h: sw ? (img.naturalWidth||img.width)   : (img.naturalHeight||img.height),
    };
}

function parseEncoded(raw) {
    if (!raw) return null;
    let filePath=raw, cropX=0, cropY=0, cropW=null, cropH=null,
        rotation=0, flipH=false, flipV=false, posX=0, posY=0;
    if (raw.includes("|")) {
        const [p,c] = raw.split("|",2);
        filePath = p;
        const n = c.split(",").map(Number);
        [cropX,cropY,cropW,cropH] = [n[0]??0, n[1]??0, n[2]??null, n[3]??null];
        rotation=n[4]??0; flipH=!!n[5]; flipV=!!n[6]; posX=n[7]??0; posY=n[8]??0;
    }
    return { filePath, cropX, cropY, cropW, cropH, rotation, flipH, flipV, posX, posY };
}

function encodeState(s) {
    if (!s.filePath) return "";
    return `${s.filePath}|${s.cropX},${s.cropY},${s.cropW},${s.cropH},${s.rotation},${s.flipH?1:0},${s.flipV?1:0},${Math.round(s.posX)},${Math.round(s.posY)}`;
}

function loadImageFromPath(filePath, cb) {
    if (!filePath) return;
    const fname = filePath.split("/").pop().split("\\").pop();
    const sub   = filePath.includes("/") ? filePath.substring(0, filePath.lastIndexOf("/")) : "";
    const url   = `/view?filename=${encodeURIComponent(fname)}&subfolder=${encodeURIComponent(sub)}&type=input`;
    const img   = new Image();
    img.onload  = () => cb(img, url);
    img.onerror = () => console.warn("ILU-Multi: could not load", url);
    img.src = url;
}

function btn(ctx, x, y, w, h, bg, border, textCol, label, fontSize) {
    ctx.fillStyle = bg;   ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = textCol;
    ctx.font = `bold ${fontSize||14}px Arial`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x + w/2, y + h/2);
}

function dualBtns(ctx, margin, bw, bh, y, lBg, rBg, lBorder, rBorder, lCol, rCol, lLabel, rLabel) {
    btn(ctx, margin,        y, bw, bh, lBg, lBorder, lCol, lLabel, 13);
    btn(ctx, margin+bw+8,   y, bw, bh, rBg, rBorder, rCol, rLabel, 13);
}

function ghostRow(ctx, margin, bw, bh, y, lLabel, rLabel) {
    dualBtns(ctx, margin, bw, bh, y, "#252525","#252525","#333","#333","#444","#444", lLabel, rLabel);
}

function placeholder(ctx, margin, W, startY, H, label) {
    ctx.fillStyle="#222"; ctx.fillRect(margin,startY,W,H);
    ctx.strokeStyle="#3a3a3a"; ctx.lineWidth=1; ctx.strokeRect(margin,startY,W,H);
    const px=margin+W*0.18, py=startY+H*0.18, pw=W*0.64, ph=H*0.64;
    ctx.fillStyle="rgba(0,0,0,0.4)";
    ctx.fillRect(margin,startY,W,H*0.18); ctx.fillRect(margin,py+ph,W,H*0.18);
    ctx.fillRect(margin,py,W*0.18,ph);    ctx.fillRect(px+pw,py,W*0.18,ph);
    ctx.save(); ctx.setLineDash([7,4]);
    ctx.strokeStyle="rgba(255,255,255,0.25)"; ctx.lineWidth=2;
    ctx.strokeRect(px,py,pw,ph); ctx.restore();
    ctx.fillStyle="#555"; ctx.font="13px Arial";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(label, margin+W/2, startY+H/2-10);
    ctx.fillStyle="#444"; ctx.font="11px Arial";
    ctx.fillText("Click Select to load an image", margin+W/2, startY+H/2+12);
}

function cropOverlay(ctx, s, W, H, bx, by, scale, ox, oy) {
    const dCX=ox+s.cropX*scale, dCY=oy+s.cropY*scale;
    const dCW=s.cropW*scale,    dCH=s.cropH*scale;
    ctx.fillStyle="rgba(0,0,0,0.55)";
    ctx.fillRect(bx,by,W,dCY-by);
    ctx.fillRect(bx,dCY+dCH,W,by+H-(dCY+dCH));
    ctx.fillRect(bx,dCY,dCX-bx,dCH);
    ctx.fillRect(dCX+dCW,dCY,bx+W-(dCX+dCW),dCH);
    ctx.strokeStyle="#fff"; ctx.lineWidth=2; ctx.strokeRect(dCX,dCY,dCW,dCH);
    ctx.save(); ctx.strokeStyle="rgba(255,255,255,0.18)"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    for (let i=1;i<=2;i++) {
        const gx=dCX+dCW*(i/3), gy=dCY+dCH*(i/3);
        ctx.beginPath(); ctx.moveTo(gx,dCY); ctx.lineTo(gx,dCY+dCH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(dCX,gy); ctx.lineTo(dCX+dCW,gy); ctx.stroke();
    }
    ctx.restore();
    [[dCX,dCY],[dCX+dCW,dCY],[dCX,dCY+dCH],[dCX+dCW,dCY+dCH]].forEach(([hx,hy])=>{
        ctx.fillStyle="rgba(255,255,255,0.95)"; ctx.strokeStyle="#333"; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.rect(hx-6,hy-6,13,13); ctx.fill(); ctx.stroke();
    });
    [[dCX+dCW/2,dCY],[dCX+dCW/2,dCY+dCH],[dCX,dCY+dCH/2],[dCX+dCW,dCY+dCH/2]].forEach(([hx,hy])=>{
        ctx.fillStyle="rgba(90,150,255,0.92)"; ctx.strokeStyle="#223"; ctx.lineWidth=1;
        ctx.beginPath(); ctx.rect(hx-4,hy-4,9,9); ctx.fill(); ctx.stroke();
    });
}

// ── Extension ─────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "ImageLoaderUltimateMulti",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageLoaderUltimateMulti") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated?.apply(this, arguments);
            this._stage = 0;
            this._imgs  = [0,1,2].map(() => ({
                filePath:null, fileName:null, el:null,
                rotation:0, flipH:false, flipV:false,
                cropX:0, cropY:0, cropW:100, cropH:100, posX:0, posY:0
            }));
            this._drag = { type:null, startX:0, startY:0, crop:null, pos:null };
            this._removeBg = false;   // mirrors the remove_background widget

            // ── Layout cache ─────────────────────────────────────────────────
            // Written ONLY inside onDrawForeground. Read ONLY inside mouse handlers.
            // Never recalculate positions in mouse handlers — always read from here.
            this._L = {
                margin: 10,
                tabY:0, tabH:28, tabW:0,           // tab strip
                selBtnY:0, selBtnH:36,              // select button
                boxY:0, BOX_W:0, BOX_H:300,         // image display box
                ox:0, oy:0, ow:0, oh:0, scale:1,    // image offset/size/scale inside box
                rotBtnY:0, flipBtnY:0,              // rotate / flip rows
                removeBgBtnY:0, removeBgBtnH:34,    // remove-background toggle
                nextBtnY:0, nextBtnH:36,            // "done" button
                bw:0, bh:34,                        // half-width button dims
            };

            this._w1 = this.widgets?.find(w=>w.name==="img1_data");
            this._w2 = this.widgets?.find(w=>w.name==="img2_data");
            this._wB = this.widgets?.find(w=>w.name==="bg_data");
            this._wRB = this.widgets?.find(w=>w.name==="remove_background");
            this.serialize_widgets = true;
            return r;
        };

        // ── Restore on page load ──────────────────────────────────────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(config) {
            const r = onConfigure?.apply(this, arguments);
            this._w1  = this.widgets?.find(w=>w.name==="img1_data");
            this._w2  = this.widgets?.find(w=>w.name==="img2_data");
            this._wB  = this.widgets?.find(w=>w.name==="bg_data");
            this._wRB = this.widgets?.find(w=>w.name==="remove_background");
            // Restore toggle state from widget
            if (this._wRB) this._removeBg = !!this._wRB.value;
            let loaded = 0;
            [this._w1, this._w2, this._wB].forEach((w,i) => {
                const p = parseEncoded(w?.value);
                if (!p?.filePath) return;
                const s = this._imgs[i];
                Object.assign(s, { filePath:p.filePath, fileName:p.filePath.split("/").pop().split("\\").pop(),
                    cropX:p.cropX, cropY:p.cropY, cropW:p.cropW??100, cropH:p.cropH??100,
                    rotation:p.rotation, flipH:p.flipH, flipV:p.flipV, posX:p.posX, posY:p.posY });
                loadImageFromPath(s.filePath, img => {
                    s.el = img;
                    const {w:sw,h:sh} = rotatedDims(img, s.rotation);
                    if (s.cropW===100 && s.cropH===100) { s.cropW=sw; s.cropH=sh; }
                    loaded++;
                    if (loaded>=3) this._stage=2;
                    else if (loaded>=2 && !this._imgs[2].filePath) this._stage=1;
                    this.setDirtyCanvas(true,true);
                });
            });
            return r;
        };

        // ── Draw ─────────────────────────────────────────────────────────────
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (this.flags.collapsed) return;
            onDrawForeground?.apply(this, arguments);

            const margin   = 10;
            const W        = this.size[0] - margin*2;
            const TITLE_H  = LiteGraph.NODE_TITLE_HEIGHT  ?? 30;
            const WIDGET_H = LiteGraph.NODE_WIDGET_HEIGHT ?? 20;
            const nW       = this.widgets?.length ?? 0;
            const wBottom  = TITLE_H + nW*(WIDGET_H+4);

            const L    = this._L;
            const stg  = this._stage;
            const BOX_H = L.BOX_H;
            const bh    = L.bh;

            // Write layout values — these are the single source of truth
            L.margin = margin;
            L.BOX_W  = W;
            L.bw     = (W-8)/2;

            // ── Tabs ─────────────────────────────────────────────────────────
            const tabY = wBottom + 95;
            const tabH = 28;
            const tabW = (W-8)/3;
            L.tabY = tabY; L.tabH = tabH; L.tabW = tabW;

            ["① Crop Image 1","② Crop Image 2","③ Compose"].forEach((label,i) => {
                const tx = margin + i*(tabW+4);
                const active = i===stg;
                ctx.fillStyle   = active ? "#4a7ac7" : "#2a2a2a";
                ctx.fillRect(tx, tabY, tabW, tabH);
                ctx.strokeStyle = active ? "#6a9ae7" : "#3a3a3a";
                ctx.lineWidth   = 1; ctx.strokeRect(tx, tabY, tabW, tabH);
                ctx.fillStyle   = active ? "#fff" : "#666";
                ctx.font        = (active?"bold ":"")+"11px Arial";
                ctx.textAlign   = "center"; ctx.textBaseline = "middle";
                ctx.fillText(label, tx+tabW/2, tabY+tabH/2);
            });

            // ── Select button ─────────────────────────────────────────────────
            const selBtnY = tabY + tabH + 8;
            L.selBtnY = selBtnY; L.selBtnH = 36;
            btn(ctx, margin, selBtnY, W, 36, "#3d6db5","#5a8fd4","#fff",
                ["📁 Select Image 1","📁 Select Image 2","📁 Select Background"][stg]);

            // ── Image box ─────────────────────────────────────────────────────
            const boxY = selBtnY + 36 + 10;
            L.boxY = boxY;
            ctx.fillStyle = "#111"; ctx.fillRect(margin, boxY, W, BOX_H);

            if (stg < 2) this._drawCropStage(ctx, stg);
            else         this._drawComposeStage(ctx);
        };

        // ── Crop stage ────────────────────────────────────────────────────────
        nodeType.prototype._drawCropStage = function(ctx, stg) {
            const L      = this._L;
            const margin = L.margin;
            const W      = L.BOX_W;
            const BOX_H  = L.BOX_H;
            const boxY   = L.boxY;
            const bw     = L.bw;
            const bh     = L.bh;
            const s      = this._imgs[stg];

            if (s.el) {
                const {w:srcW, h:srcH} = rotatedDims(s.el, s.rotation);
                const scale = Math.min(W/srcW, BOX_H/srcH);
                const dw = srcW*scale, dh = srcH*scale;
                const ox = margin + (W-dw)/2;
                const oy = boxY   + (BOX_H-dh)/2;
                L.scale=scale; L.ox=ox; L.oy=oy; L.ow=dw; L.oh=dh;

                ctx.drawImage(getRotatedCanvas(s.el,s.rotation,s.flipH,s.flipV), ox, oy, dw, dh);
                cropOverlay(ctx, s, W, BOX_H, margin, boxY, scale, ox, oy);

                const infoY = boxY + BOX_H + 7;
                ctx.fillStyle="#888"; ctx.font="11px Arial";
                ctx.textAlign="left"; ctx.textBaseline="top";
                ctx.fillText(`Crop: ${s.cropW}×${s.cropH}px   Pos: (${s.cropX}, ${s.cropY})   Rot: ${s.rotation}°`, margin, infoY);

                const rotBtnY = infoY + 18;
                L.rotBtnY = rotBtnY;
                dualBtns(ctx,margin,bw,bh,rotBtnY, "#303040","#303040","#505068","#505068","#bbc","#bbc","↺ Rotate CCW","↻ Rotate CW");

                const flipBtnY = rotBtnY + bh + 6;
                L.flipBtnY = flipBtnY;
                dualBtns(ctx,margin,bw,bh,flipBtnY,
                    s.flipH?"#3a5a3a":"#303040", s.flipV?"#3a5a3a":"#303040",
                    s.flipH?"#5a9a5a":"#505068", s.flipV?"#5a9a5a":"#505068",
                    s.flipH?"#8f8":"#bbc",        s.flipV?"#8f8":"#bbc",
                    s.flipH?"⇔ Flip H ✓":"⇔ Flip H", s.flipV?"⇕ Flip V ✓":"⇕ Flip V");

                const nextY = flipBtnY + bh + 10;
                L.nextBtnY = nextY; L.nextBtnH = 36;
                btn(ctx, margin, nextY, W, 36, "#2a5a2a","#4a9a4a","#8f8",
                    stg===0 ? "✓ Done — Crop Image 2 →" : "✓ Done — Set Background →");

                const bottom = nextY + 36 + 10;
                if (Math.abs(this.size[1]-bottom)>2) { this.size[1]=bottom; this.setDirtyCanvas(true,true); }
            } else {
                L.rotBtnY=0; L.flipBtnY=0; L.nextBtnY=0;
                placeholder(ctx, margin, W, boxY, BOX_H, stg===0?"No Image 1 — select above":"No Image 2 — select above");
                const gy1 = boxY+BOX_H+26, gy2 = gy1+bh+6;
                ghostRow(ctx,margin,bw,bh,gy1,"↺ Rotate CCW","↻ Rotate CW");
                ghostRow(ctx,margin,bw,bh,gy2,"⇔ Flip H","⇕ Flip V");
                const bottom = gy2+bh+10;
                if (Math.abs(this.size[1]-bottom)>2) { this.size[1]=bottom; this.setDirtyCanvas(true,true); }
            }
        };

        // ── Compose stage ─────────────────────────────────────────────────────
        nodeType.prototype._drawComposeStage = function(ctx) {
            const L      = this._L;
            const margin = L.margin;
            const W      = L.BOX_W;
            const BOX_H  = L.BOX_H;
            const boxY   = L.boxY;
            const bw     = L.bw;
            const bh     = L.bh;
            const bg     = this._imgs[2];

            if (bg.el) {
                const bgDims  = rotatedDims(bg.el, bg.rotation);
                const scale   = Math.min(W/bgDims.w, BOX_H/bgDims.h);
                const bgW     = bgDims.w*scale, bgH = bgDims.h*scale;
                const bgOx    = margin + (W-bgW)/2;
                const bgOy    = boxY   + (BOX_H-bgH)/2;
                L.scale=scale; L.ox=bgOx; L.oy=bgOy; L.ow=bgW; L.oh=bgH;

                ctx.drawImage(getRotatedCanvas(bg.el,bg.rotation,bg.flipH,bg.flipV), bgOx, bgOy, bgW, bgH);

                // Mirror Python's resize logic so preview matches final output:
                // landscape bg → crop height matches bg height
                // portrait/square bg → crop height = bg height / 2
                const bgIsLandscape = bgDims.w > bgDims.h;

                [[this._imgs[0],"#4af","1"],[this._imgs[1],"#fa4","2"]].forEach(([s,color,label])=>{
                    if (!s.el) return;
                    const rc = getRotatedCanvas(s.el,s.rotation,s.flipH,s.flipV);
                    const cc = document.createElement("canvas");
                    cc.width=s.cropW; cc.height=s.cropH;
                    cc.getContext("2d").drawImage(rc,-s.cropX,-s.cropY);

                    // Compute resize scale matching Python's resize_to_bg()
                    const targetH = bgIsLandscape ? bgDims.h : bgDims.h / 2;
                    const resizeScale = targetH / s.cropH;
                    const resizedW = s.cropW * resizeScale;
                    const resizedH = s.cropH * resizeScale;

                    // Draw at display scale
                    const dx=bgOx+s.posX*scale, dy=bgOy+s.posY*scale;
                    const dw=resizedW*scale,     dh=resizedH*scale;
                    ctx.drawImage(cc,dx,dy,dw,dh);
                    ctx.strokeStyle=color; ctx.lineWidth=2;
                    ctx.setLineDash([4,3]); ctx.strokeRect(dx,dy,dw,dh); ctx.setLineDash([]);
                    ctx.fillStyle=color;
                    ctx.beginPath(); ctx.arc(dx+8,dy+8,8,0,Math.PI*2); ctx.fill();
                    ctx.fillStyle="#000"; ctx.font="bold 10px Arial";
                    ctx.textAlign="center"; ctx.textBaseline="middle";
                    ctx.fillText(label,dx+8,dy+8);
                });

                const infoY = boxY+BOX_H+7;
                ctx.fillStyle="#888"; ctx.font="11px Arial";
                ctx.textAlign="left"; ctx.textBaseline="top";
                ctx.fillText("🔵 Drag ① or ② to reposition on background", margin, infoY);

                const rotBtnY = infoY+18;
                L.rotBtnY = rotBtnY;
                dualBtns(ctx,margin,bw,bh,rotBtnY,"#303040","#303040","#505068","#505068","#bbc","#bbc","↺ BG Rotate CCW","↻ BG Rotate CW");

                const flipBtnY = rotBtnY+bh+6;
                L.flipBtnY = flipBtnY;
                dualBtns(ctx,margin,bw,bh,flipBtnY,
                    bg.flipH?"#3a5a3a":"#303040", bg.flipV?"#3a5a3a":"#303040",
                    bg.flipH?"#5a9a5a":"#505068", bg.flipV?"#5a9a5a":"#505068",
                    bg.flipH?"#8f8":"#bbc",        bg.flipV?"#8f8":"#bbc",
                    bg.flipH?"⇔ BG Flip H ✓":"⇔ BG Flip H", bg.flipV?"⇕ BG Flip V ✓":"⇕ BG Flip V");

                // Remove background toggle — full width
                const removeBgBtnY = flipBtnY + bh + 8;
                const removeBgBtnH = L.removeBgBtnH;
                L.removeBgBtnY = removeBgBtnY;
                const rbActive = this._removeBg;
                btn(ctx, margin, removeBgBtnY, W, removeBgBtnH,
                    rbActive ? "#4a2a6a" : "#2a2a3a",
                    rbActive ? "#8a5aaa" : "#404058",
                    rbActive ? "#d8a0ff" : "#778",
                    rbActive ? "✂️ Remove Background: ON" : "✂️ Remove Background: OFF");

                const bottom = removeBgBtnY + removeBgBtnH + 10;
                if (Math.abs(this.size[1]-bottom)>2) { this.size[1]=bottom; this.setDirtyCanvas(true,true); }
            } else {
                L.rotBtnY=0; L.flipBtnY=0; L.removeBgBtnY=0;
                placeholder(ctx, margin, W, boxY, BOX_H, "No background — select above");
                const gy1=boxY+BOX_H+26, gy2=gy1+bh+6;
                ghostRow(ctx,margin,bw,bh,gy1,"↺ BG Rotate CCW","↻ BG Rotate CW");
                ghostRow(ctx,margin,bw,bh,gy2,"⇔ BG Flip H","⇕ BG Flip V");
                btn(ctx, margin, gy2+bh+8, W, L.removeBgBtnH, "#1a1a22","#303038","#445","✂️ Remove Background: OFF");
                const bottom=gy2+bh+8+L.removeBgBtnH+10;
                if (Math.abs(this.size[1]-bottom)>2) { this.size[1]=bottom; this.setDirtyCanvas(true,true); }
            }
        };

        // ── Mouse Down ────────────────────────────────────────────────────────
        // Reads ONLY from this._L — no position recalculation here.
        nodeType.prototype.onMouseDown = function(e, pos, canvas) {
            const L   = this._L;
            const m   = L.margin;
            const W   = this.size[0] - m*2;
            const stg = this._stage;
            const bw  = L.bw, bh = L.bh;
            const x   = pos[0], y = pos[1];

            // Tabs
            if (y >= L.tabY && y <= L.tabY + L.tabH) {
                for (let i=0; i<3; i++) {
                    const tx = m + i*(L.tabW+4);
                    if (x >= tx && x <= tx+L.tabW) { this._stage=i; canvas.setDirty(true); return true; }
                }
            }

            // Select button
            if (y >= L.selBtnY && y <= L.selBtnY+L.selBtnH && x >= m && x <= m+W) {
                this._openFilePicker(stg, canvas); return true;
            }

            // Next button
            if (stg<2 && L.nextBtnY>0 && this._imgs[stg].el &&
                y >= L.nextBtnY && y <= L.nextBtnY+L.nextBtnH && x >= m && x <= m+W) {
                this._stage = stg+1; canvas.setDirty(true); return true;
            }

            // Rotate buttons
            if (L.rotBtnY>0 && y >= L.rotBtnY && y <= L.rotBtnY+bh) {
                const idx = stg===2 ? 2 : stg;
                if (x >= m && x <= m+bw)         { this._doRotate(idx,-1); canvas.setDirty(true); return true; }
                if (x >= m+bw+8 && x <= m+W)     { this._doRotate(idx,+1); canvas.setDirty(true); return true; }
            }

            // Flip buttons
            if (L.flipBtnY>0 && y >= L.flipBtnY && y <= L.flipBtnY+bh) {
                const idx = stg===2 ? 2 : stg;
                const s = this._imgs[idx];
                if (!s.el) return false;
                const {w:srcW, h:srcH} = rotatedDims(s.el, s.rotation);
                if (x >= m && x <= m+bw)     { s.cropX=srcW-s.cropX-s.cropW; s.flipH=!s.flipH; this._saveAll(); canvas.setDirty(true); return true; }
                if (x >= m+bw+8 && x <= m+W) { s.cropY=srcH-s.cropY-s.cropH; s.flipV=!s.flipV; this._saveAll(); canvas.setDirty(true); return true; }
            }

            // Remove background toggle (compose stage only)
            if (stg===2 && L.removeBgBtnY>0 &&
                y >= L.removeBgBtnY && y <= L.removeBgBtnY + L.removeBgBtnH &&
                x >= m && x <= m+W) {
                this._removeBg = !this._removeBg;
                if (this._wRB) this._wRB.value = this._removeBg;
                canvas.setDirty(true);
                return true;
            }

            // Crop handles
            if (stg<2 && this._imgs[stg].el) {
                const s = this._imgs[stg];
                const {w:srcW,h:srcH} = rotatedDims(s.el, s.rotation);
                const {scale:sc, ox, oy, ow, oh} = L;
                if (x>=ox && x<=ox+ow && y>=oy && y<=oy+oh) {
                    const dCX=ox+s.cropX*sc, dCY=oy+s.cropY*sc;
                    const dCW=s.cropW*sc,     dCH=s.cropH*sc;
                    const hs=14;
                    const handles=[
                        {name:'top-left',     x:dCX,       y:dCY      },
                        {name:'top-right',    x:dCX+dCW,   y:dCY      },
                        {name:'bottom-left',  x:dCX,       y:dCY+dCH  },
                        {name:'bottom-right', x:dCX+dCW,   y:dCY+dCH  },
                        {name:'top',          x:dCX+dCW/2, y:dCY      },
                        {name:'bottom',       x:dCX+dCW/2, y:dCY+dCH  },
                        {name:'left',         x:dCX,       y:dCY+dCH/2},
                        {name:'right',        x:dCX+dCW,   y:dCY+dCH/2},
                    ];
                    for (const h of handles) {
                        if (Math.abs(x-h.x)<hs && Math.abs(y-h.y)<hs) {
                            this._drag = {type:h.name, startX:x, startY:y, crop:{x:s.cropX,y:s.cropY,w:s.cropW,h:s.cropH,srcW,srcH}};
                            return true;
                        }
                    }
                    if (x>=dCX && x<=dCX+dCW && y>=dCY && y<=dCY+dCH) {
                        this._drag = {type:'move', startX:x, startY:y, crop:{x:s.cropX,y:s.cropY,w:s.cropW,h:s.cropH,srcW,srcH}};
                        return true;
                    }
                }
            }

            // Overlay position drag
            if (stg===2 && this._imgs[2].el) {
                const {scale:sc, ox:bgOx, oy:bgOy} = L;
                const bgDims = rotatedDims(this._imgs[2].el, this._imgs[2].rotation);
                const bgIsLandscape = bgDims.w > bgDims.h;
                for (const [idx,key] of [[0,'pos1'],[1,'pos2']]) {
                    const s = this._imgs[idx];
                    if (!s.el) continue;
                    // Match Python resize: landscape → bg height, portrait → bg height / 2
                    const targetH    = bgIsLandscape ? bgDims.h : bgDims.h / 2;
                    const resizeScale = targetH / s.cropH;
                    const resizedW   = s.cropW * resizeScale;
                    const resizedH   = s.cropH * resizeScale;
                    const dx=bgOx+s.posX*sc, dy=bgOy+s.posY*sc;
                    const dw=resizedW*sc,     dh=resizedH*sc;
                    if (x>=dx && x<=dx+dw && y>=dy && y<=dy+dh) {
                        this._drag = {type:key, startX:x, startY:y, pos:{x:s.posX,y:s.posY,bgW:bgDims.w,bgH:bgDims.h}};
                        return true;
                    }
                }
            }

            return false;
        };

        // ── Mouse Move ────────────────────────────────────────────────────────
        nodeType.prototype.onMouseMove = function(e, pos, canvas) {
            const d = this._drag;
            if (!d?.type) return;
            const sc = this._L.scale;
            const x=pos[0], y=pos[1];
            const dx=(x-d.startX)/sc, dy=(y-d.startY)/sc;

            if (d.type==='pos1' || d.type==='pos2') {
                const s = this._imgs[d.type==='pos1'?0:1];
                s.posX = Math.max(0, Math.min(Math.round(d.pos.x+dx), d.pos.bgW-s.cropW));
                s.posY = Math.max(0, Math.min(Math.round(d.pos.y+dy), d.pos.bgH-s.cropH));
                this._saveAll(); canvas.setDirty(true); return;
            }

            const s  = this._imgs[this._stage];
            const dc = d.crop;
            const mn = 10;
            let nx=dc.x, ny=dc.y, nw=dc.w, nh=dc.h;
            switch (d.type) {
                case 'move':         nx=Math.max(0,Math.min(dc.x+dx,dc.srcW-dc.w));    ny=Math.max(0,Math.min(dc.y+dy,dc.srcH-dc.h)); break;
                case 'top':          ny=Math.max(0,Math.min(dc.y+dy,dc.y+dc.h-mn));    nh=dc.h-(ny-dc.y); break;
                case 'bottom':       nh=Math.max(mn,Math.min(dc.h+dy,dc.srcH-dc.y));   break;
                case 'left':         nx=Math.max(0,Math.min(dc.x+dx,dc.x+dc.w-mn));    nw=dc.w-(nx-dc.x); break;
                case 'right':        nw=Math.max(mn,Math.min(dc.w+dx,dc.srcW-dc.x));   break;
                case 'top-left':     ny=Math.max(0,Math.min(dc.y+dy,dc.y+dc.h-mn));    nh=dc.h-(ny-dc.y); nx=Math.max(0,Math.min(dc.x+dx,dc.x+dc.w-mn)); nw=dc.w-(nx-dc.x); break;
                case 'top-right':    ny=Math.max(0,Math.min(dc.y+dy,dc.y+dc.h-mn));    nh=dc.h-(ny-dc.y); nw=Math.max(mn,Math.min(dc.w+dx,dc.srcW-dc.x)); break;
                case 'bottom-left':  nh=Math.max(mn,Math.min(dc.h+dy,dc.srcH-dc.y));   nx=Math.max(0,Math.min(dc.x+dx,dc.x+dc.w-mn)); nw=dc.w-(nx-dc.x); break;
                case 'bottom-right': nw=Math.max(mn,Math.min(dc.w+dx,dc.srcW-dc.x));   nh=Math.max(mn,Math.min(dc.h+dy,dc.srcH-dc.y)); break;
            }
            s.cropX=Math.max(0,Math.min(Math.round(nx),dc.srcW-mn));
            s.cropY=Math.max(0,Math.min(Math.round(ny),dc.srcH-mn));
            s.cropW=Math.max(mn,Math.min(Math.round(nw),dc.srcW-s.cropX));
            s.cropH=Math.max(mn,Math.min(Math.round(nh),dc.srcH-s.cropY));
            this._saveAll(); canvas.setDirty(true);
        };

        nodeType.prototype.onMouseUp = function() { this._drag = {type:null}; };

        // ── Helpers ───────────────────────────────────────────────────────────

        nodeType.prototype._openFilePicker = function(imgIdx, canvas) {
            const fi = document.createElement("input");
            fi.type="file"; fi.accept="image/*";
            fi.style.cssText="position:fixed;top:0;left:0;opacity:0;pointer-events:none;";
            fi.addEventListener("change", async ev => {
                const file = ev.target.files[0];
                if (!file) { document.body.removeChild(fi); return; }
                const s = this._imgs[imgIdx];
                s.fileName=file.name; s.rotation=0; s.flipH=false; s.flipV=false;
                const reader = new FileReader();
                reader.onload = re => {
                    const img = new Image();
                    img.onload = () => {
                        s.el=img;
                        const {w:sw,h:sh}=rotatedDims(img,0);
                        s.cropX=0; s.cropY=0; s.cropW=sw; s.cropH=sh;
                        this._saveAll(); canvas.setDirty(true);
                    };
                    img.src=re.target.result;
                };
                reader.readAsDataURL(file);
                const fd=new FormData();
                fd.append("image",file); fd.append("subfolder","ilu_multi_uploads"); fd.append("type","input");
                try {
                    const res = await fetch("/upload/image",{method:"POST",body:fd});
                    if (res.ok) { const j=await res.json(); s.filePath=j.subfolder?`${j.subfolder}/${j.name}`:j.name; this._saveAll(); }
                } catch(err) { console.error("Upload error:",err); }
                document.body.removeChild(fi);
            });
            document.body.appendChild(fi); fi.click();
        };

        nodeType.prototype._doRotate = function(idx, dir) {
            const s = this._imgs[idx];
            if (!s.el) return;
            const {w:oldW,h:oldH} = rotatedDims(s.el,s.rotation);
            const [ox,oy,ow,oh] = [s.cropX,s.cropY,s.cropW,s.cropH];
            s.rotation=(s.rotation+dir*90+360)%360;
            const {w:nW,h:nH} = rotatedDims(s.el,s.rotation);
            if (dir===1) { s.cropX=oldH-oy-oh; s.cropY=ox; s.cropW=oh; s.cropH=ow; }
            else         { s.cropX=oy; s.cropY=oldW-ox-ow; s.cropW=oh; s.cropH=ow; }
            s.cropX=Math.max(0,Math.min(s.cropX,nW-1));
            s.cropY=Math.max(0,Math.min(s.cropY,nH-1));
            s.cropW=Math.max(1,Math.min(s.cropW,nW-s.cropX));
            s.cropH=Math.max(1,Math.min(s.cropH,nH-s.cropY));
            this._saveAll();
        };

        nodeType.prototype._saveAll = function() {
            [this._w1,this._w2,this._wB].forEach((w,i) => { if(w) w.value=encodeState(this._imgs[i]); });
        };
    }
});