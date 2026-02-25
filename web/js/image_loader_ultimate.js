/**
 * ComfyUI Custom Node - Image Selector with Interactive Cropper Frontend
 * Place this file in: ComfyUI/custom_nodes/image_loader_ultimate/image_loader_ultimate.js
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "ImageLoaderUltimate",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageLoaderUltimate") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Initialize state
                this.selectedImagePath = null;
                this.selectedFileName = null;
                this.imageElement = null;
                this.imageData = null;
                this.originalWidth = 0;
                this.originalHeight = 0;
                this.rotation = 0; // 0, 90, 180, 270
                this.flipH = false; // flip horizontal
                this.flipV = false; // flip vertical
                
                // Crop state
                this.cropX = 0;
                this.cropY = 0;
                this.cropWidth = 100;
                this.cropHeight = 100;
                this.dragging = null;
                this.dragStartX = 0;
                this.dragStartY = 0;
                this.dragStartCrop = null;
                
                // UI layout state (set during draw, read during mouse events)
                this.displayScale = 1;
                this.imageDisplayWidth = 0;
                this.imageDisplayHeight = 0;
                this.imageOffsetX = 0;
                this.imageOffsetY = 0;
                this._rotateButtonY = 0;
                this._flipButtonY = 0;
                this._selectButtonY = 0;
                this._selectButtonH = 0;
                
                // Get widgets
                this.imagePathWidget = this.widgets.find(w => w.name === "image_path");
                this.cropDataWidget  = this.widgets.find(w => w.name === "crop_data");
                
                this.serialize_widgets = true;
                return result;
            };

            // ── Helpers ──────────────────────────────────────────────────────

            /** Returns a canvas element with the image rotated and flipped */
            nodeType.prototype._getRotatedCanvas = function() {
                const img = this.imageElement;
                if (!img) return null;
                const rot = this.rotation;
                const offscreen = document.createElement("canvas");
                if (rot === 90 || rot === 270) {
                    offscreen.width  = img.naturalHeight || img.height;
                    offscreen.height = img.naturalWidth  || img.width;
                } else {
                    offscreen.width  = img.naturalWidth  || img.width;
                    offscreen.height = img.naturalHeight || img.height;
                }
                const octx = offscreen.getContext("2d");
                octx.translate(offscreen.width / 2, offscreen.height / 2);
                octx.rotate((rot * Math.PI) / 180);
                // Apply flips — scale(-1,1) for H, scale(1,-1) for V
                octx.scale(this.flipH ? -1 : 1, this.flipV ? -1 : 1);
                octx.drawImage(img, -(img.naturalWidth || img.width) / 2, -(img.naturalHeight || img.height) / 2);
                return offscreen;
            };

            /** Dimensions of the image after applying current rotation */
            nodeType.prototype._rotatedDims = function() {
                const img = this.imageElement;
                if (!img) return { w: 0, h: 0 };
                const rot = this.rotation;
                const swapped = rot === 90 || rot === 270;
                return {
                    w: swapped ? (img.naturalHeight || img.height) : (img.naturalWidth  || img.width),
                    h: swapped ? (img.naturalWidth  || img.width)  : (img.naturalHeight || img.height),
                };
            };

            /** Draw a single square handle */
            nodeType.prototype._drawHandle = function(ctx, x, y, size, fillColor, strokeColor) {
                ctx.fillStyle   = fillColor;
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth   = 1.5;
                ctx.beginPath();
                ctx.rect(x - size / 2, y - size / 2, size, size);
                ctx.fill();
                ctx.stroke();
            };

            /** Draw the placeholder crop UI shown before any image is selected */
            nodeType.prototype._drawPlaceholderCropUI = function(ctx, margin, rectWidth, startY, areaH) {
                // Background
                ctx.fillStyle = "#222";
                ctx.fillRect(margin, startY, rectWidth, areaH);
                ctx.strokeStyle = "#3a3a3a";
                ctx.lineWidth = 1;
                ctx.strokeRect(margin, startY, rectWidth, areaH);

                // Placeholder crop rect (~60% inset)
                const padX = rectWidth * 0.18;
                const padY = areaH * 0.18;
                const pX = margin + padX;
                const pY = startY + padY;
                const pW = rectWidth - padX * 2;
                const pH = areaH - padY * 2;

                // Dimmed areas outside placeholder crop
                ctx.fillStyle = "rgba(0,0,0,0.4)";
                ctx.fillRect(margin, startY, rectWidth, padY);
                ctx.fillRect(margin, pY + pH, rectWidth, padY);
                ctx.fillRect(margin, pY, padX, pH);
                ctx.fillRect(pX + pW, pY, padX, pH);

                // Dashed crop border
                ctx.save();
                ctx.setLineDash([7, 4]);
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 2;
                ctx.strokeRect(pX, pY, pW, pH);
                ctx.restore();

                // Rule-of-thirds lines
                ctx.save();
                ctx.strokeStyle = "rgba(255,255,255,0.08)";
                ctx.lineWidth = 1;
                for (let i = 1; i <= 2; i++) {
                    const gx = pX + pW * (i / 3);
                    const gy = pY + pH * (i / 3);
                    ctx.beginPath(); ctx.moveTo(gx, pY); ctx.lineTo(gx, pY + pH); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(pX, gy); ctx.lineTo(pX + pW, gy); ctx.stroke();
                }
                ctx.restore();

                // Corner handles (ghosted)
                [[pX, pY], [pX + pW, pY], [pX, pY + pH], [pX + pW, pY + pH]].forEach(([hx, hy]) => {
                    this._drawHandle(ctx, hx, hy, 12, "rgba(255,255,255,0.2)", "rgba(255,255,255,0.3)");
                });
                // Edge handles (ghosted)
                [[pX + pW/2, pY], [pX + pW/2, pY + pH], [pX, pY + pH/2], [pX + pW, pY + pH/2]].forEach(([hx, hy]) => {
                    this._drawHandle(ctx, hx, hy, 9, "rgba(100,160,255,0.2)", "rgba(100,160,255,0.3)");
                });

                // Label
                ctx.fillStyle = "#555";
                ctx.font = "13px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("No image selected — click Select Image above", margin + rectWidth / 2, startY + areaH / 2 - 12);
                ctx.fillStyle = "#444";
                ctx.font = "11px Arial";
                ctx.fillText("Drag handles to adjust crop region after loading", margin + rectWidth / 2, startY + areaH / 2 + 12);
            };

            // ── Drawing ───────────────────────────────────────────────────────

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (this.flags.collapsed) return;
                const result = onDrawForeground?.apply(this, arguments);
                
                const margin = 10;
                const rectWidth = this.size[0] - margin * 2;

                // Calculate Y below all widgets reliably using LiteGraph constants
                const TITLE_H  = LiteGraph.NODE_TITLE_HEIGHT  ?? 30;
                const WIDGET_H = LiteGraph.NODE_WIDGET_HEIGHT  ?? 20;
                const WIDGET_PAD = 4;
                const nWidgets = (this.widgets?.length ?? 0);
                const widgetBottom = TITLE_H + nWidgets * (WIDGET_H + WIDGET_PAD);

                // ── File path display ──
                const pathY = widgetBottom + 18;
                this._selectButtonY = pathY + 34 + 6; // store for onMouseDown
                this._selectButtonH = 38;
                ctx.fillStyle = "#2c2c2c";
                ctx.fillRect(margin, pathY, rectWidth, 30);
                ctx.strokeStyle = "#4a4a4a";
                ctx.lineWidth = 1;
                ctx.strokeRect(margin, pathY, rectWidth, 30);
                ctx.fillStyle = "#999";
                ctx.font = "12px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                let pathText = this.selectedFileName || "No image selected";
                if (pathText.length > 44) pathText = "..." + pathText.slice(-41);
                ctx.fillText(`📄 ${pathText}`, margin + 8, pathY + 15);

                // ── Select image button ──
                const buttonY = this._selectButtonY;
                const buttonH = this._selectButtonH;
                ctx.fillStyle = "#3d6db5";
                ctx.fillRect(margin, buttonY, rectWidth, buttonH);
                ctx.strokeStyle = "#5a8fd4";
                ctx.lineWidth = 1;
                ctx.strokeRect(margin, buttonY, rectWidth, buttonH);
                ctx.fillStyle = "#fff";
                ctx.font = "bold 15px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("📁 Select Image", margin + rectWidth / 2, buttonY + buttonH / 2);

                const imageAreaStartY = buttonY + buttonH + 14;
                const rBtnH = 34;
                const rBtnW = (rectWidth - 8) / 2;

                // Track total content bottom so we can resize the node to fit
                let contentBottom;

                if (this.imageElement && this.imageData) {
                    // ── Loaded image inside a fixed display box ──
                    // The box is always the same size; image fits inside (letterboxed/pillarboxed)
                    const BOX_W = rectWidth;
                    const BOX_H = 300;
                    const boxX  = margin;
                    const boxY  = imageAreaStartY;

                    const { w: srcW, h: srcH } = this._rotatedDims();
                    this.displayScale       = Math.min(BOX_W / srcW, BOX_H / srcH);
                    this.imageDisplayWidth  = srcW * this.displayScale;
                    this.imageDisplayHeight = srcH * this.displayScale;
                    // Centre image inside the fixed box
                    this.imageOffsetX = boxX + (BOX_W - this.imageDisplayWidth)  / 2;
                    this.imageOffsetY = boxY + (BOX_H - this.imageDisplayHeight) / 2;

                    // Draw fixed box background
                    ctx.fillStyle = "#111";
                    ctx.fillRect(boxX, boxY, BOX_W, BOX_H);

                    // Draw image centred inside the box
                    const rotCanvas = this._getRotatedCanvas();
                    ctx.drawImage(rotCanvas, this.imageOffsetX, this.imageOffsetY,
                                  this.imageDisplayWidth, this.imageDisplayHeight);

                    // Clamp crop to rotated image size
                    const cropX = Math.max(0, Math.min(this.cropX, srcW - 1));
                    const cropY = Math.max(0, Math.min(this.cropY, srcH - 1));
                    const cropW = Math.max(1, Math.min(this.cropWidth,  srcW - cropX));
                    const cropH = Math.max(1, Math.min(this.cropHeight, srcH - cropY));

                    const dCX = this.imageOffsetX + cropX * this.displayScale;
                    const dCY = this.imageOffsetY + cropY * this.displayScale;
                    const dCW = cropW * this.displayScale;
                    const dCH = cropH * this.displayScale;

                    // Dim the entire box, then cut out the crop window
                    // This also darkens any letterbox/pillarbox bars around the image
                    ctx.fillStyle = "rgba(0,0,0,0.55)";
                    ctx.fillRect(boxX, boxY, BOX_W, dCY - boxY);                              // above crop
                    ctx.fillRect(boxX, dCY + dCH, BOX_W, boxY + BOX_H - (dCY + dCH));        // below crop
                    ctx.fillRect(boxX, dCY, dCX - boxX, dCH);                                 // left of crop
                    ctx.fillRect(dCX + dCW, dCY, boxX + BOX_W - (dCX + dCW), dCH);           // right of crop

                    // Crop border
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(dCX, dCY, dCW, dCH);

                    // Rule-of-thirds grid inside crop
                    ctx.save();
                    ctx.strokeStyle = "rgba(255,255,255,0.18)";
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    for (let i = 1; i <= 2; i++) {
                        const gx = dCX + dCW * (i / 3);
                        const gy = dCY + dCH * (i / 3);
                        ctx.beginPath(); ctx.moveTo(gx, dCY); ctx.lineTo(gx, dCY + dCH); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(dCX, gy); ctx.lineTo(dCX + dCW, gy); ctx.stroke();
                    }
                    ctx.restore();

                    // Corner handles — white, larger
                    [[dCX, dCY], [dCX + dCW, dCY], [dCX, dCY + dCH], [dCX + dCW, dCY + dCH]].forEach(([hx, hy]) => {
                        this._drawHandle(ctx, hx, hy, 13, "rgba(255,255,255,0.95)", "#333");
                    });
                    // Edge handles — blue accent, smaller
                    [[dCX + dCW/2, dCY], [dCX + dCW/2, dCY + dCH],
                     [dCX, dCY + dCH/2], [dCX + dCW, dCY + dCH/2]].forEach(([hx, hy]) => {
                        this._drawHandle(ctx, hx, hy, 9, "rgba(90,150,255,0.92)", "#223");
                    });

                    // Info strip — anchored to fixed box bottom, not image bottom
                    const infoY = boxY + BOX_H + 7;
                    ctx.fillStyle = "#888";
                    ctx.font = "11px Arial";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "top";
                    ctx.fillText(
                        `Crop: ${cropW} × ${cropH} px   Position: (${cropX}, ${cropY})   Rotation: ${this.rotation}°`,
                        margin, infoY
                    );

                    // ── Rotate buttons ──
                    const rBtnY = infoY + 20;
                    this._rotateButtonY = rBtnY;

                    // CCW
                    ctx.fillStyle = "#303040";
                    ctx.fillRect(margin, rBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = "#505068";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(margin, rBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = "#bbc";
                    ctx.font = "bold 13px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("↺  Rotate CCW", margin + rBtnW / 2, rBtnY + rBtnH / 2);

                    // CW
                    ctx.fillStyle = "#303040";
                    ctx.fillRect(margin + rBtnW + 8, rBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = "#505068";
                    ctx.strokeRect(margin + rBtnW + 8, rBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = "#bbc";
                    ctx.fillText("↻  Rotate CW", margin + rBtnW + 8 + rBtnW / 2, rBtnY + rBtnH / 2);

                    // ── Flip buttons ──
                    const fBtnY = rBtnY + rBtnH + 6;
                    this._flipButtonY = fBtnY;

                    // Flip H — highlight if active
                    ctx.fillStyle = this.flipH ? "#3a5a3a" : "#303040";
                    ctx.fillRect(margin, fBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = this.flipH ? "#5a9a5a" : "#505068";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(margin, fBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = this.flipH ? "#8f8" : "#bbc";
                    ctx.font = "bold 13px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("⇔  Flip H", margin + rBtnW / 2, fBtnY + rBtnH / 2);

                    // Flip V — highlight if active
                    ctx.fillStyle = this.flipV ? "#3a5a3a" : "#303040";
                    ctx.fillRect(margin + rBtnW + 8, fBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = this.flipV ? "#5a9a5a" : "#505068";
                    ctx.strokeRect(margin + rBtnW + 8, fBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = this.flipV ? "#8f8" : "#bbc";
                    ctx.fillText("⇕  Flip V", margin + rBtnW + 8 + rBtnW / 2, fBtnY + rBtnH / 2);

                    contentBottom = fBtnY + rBtnH + 10;

                } else {
                    // ── Placeholder UI ──
                    const placeholderH = 220;
                    this._drawPlaceholderCropUI(ctx, margin, rectWidth, imageAreaStartY, placeholderH);
                    this._rotateButtonY = 0; // no active rotate buttons

                    // Greyed-out rotate buttons
                    const rBtnY = imageAreaStartY + placeholderH + 12;
                    ctx.fillStyle = "#252525";
                    ctx.fillRect(margin, rBtnY, rBtnW, rBtnH);
                    ctx.fillRect(margin + rBtnW + 8, rBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(margin, rBtnY, rBtnW, rBtnH);
                    ctx.strokeRect(margin + rBtnW + 8, rBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = "#444";
                    ctx.font = "bold 13px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("↺  Rotate CCW", margin + rBtnW / 2, rBtnY + rBtnH / 2);
                    ctx.fillText("↻  Rotate CW", margin + rBtnW + 8 + rBtnW / 2, rBtnY + rBtnH / 2);

                    // Greyed-out flip buttons
                    const fBtnY = rBtnY + rBtnH + 6;
                    this._flipButtonY = 0;
                    ctx.fillStyle = "#252525";
                    ctx.fillRect(margin, fBtnY, rBtnW, rBtnH);
                    ctx.fillRect(margin + rBtnW + 8, fBtnY, rBtnW, rBtnH);
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(margin, fBtnY, rBtnW, rBtnH);
                    ctx.strokeRect(margin + rBtnW + 8, fBtnY, rBtnW, rBtnH);
                    ctx.fillStyle = "#444";
                    ctx.fillText("⇔  Flip H", margin + rBtnW / 2, fBtnY + rBtnH / 2);
                    ctx.fillText("⇕  Flip V", margin + rBtnW + 8 + rBtnW / 2, fBtnY + rBtnH / 2);

                    contentBottom = fBtnY + rBtnH + 10;
                }

                // Resize node height to fit all drawn content
                if (contentBottom && Math.abs(this.size[1] - contentBottom) > 2) {
                    this.size[1] = contentBottom;
                    this.setDirtyCanvas(true, true);
                }

                return result;
            };

            // ── Mouse handling ────────────────────────────────────────────────

            nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
                const margin = 10;
                const rectWidth = this.size[0] - margin * 2;

                // Use positions stored during onDrawForeground — these are always accurate
                const buttonY = this._selectButtonY;
                const buttonH = this._selectButtonH;
                const rBtnW = (rectWidth - 8) / 2;
                const rBtnH = 34;

                // ── Select image button ──
                if (localPos[0] >= margin && localPos[0] <= margin + rectWidth &&
                    localPos[1] >= buttonY && localPos[1] <= buttonY + buttonH) {

                    const fileInput = document.createElement("input");
                    fileInput.type = "file";
                    fileInput.accept = "image/*";
                    fileInput.style.cssText = "position:fixed;top:0;left:0;opacity:0;pointer-events:none;";

                    fileInput.addEventListener("change", async (ev) => {
                        const file = ev.target.files[0];
                        if (!file) { document.body.removeChild(fileInput); return; }

                        this.selectedFileName = file.name;
                        this.rotation = 0;
                        this.flipH = false;
                        this.flipV = false;

                        // Preview
                        const reader = new FileReader();
                        reader.onload = (re) => {
                            const img = new Image();
                            img.onload = () => {
                                this.originalWidth  = img.naturalWidth  || img.width;
                                this.originalHeight = img.naturalHeight || img.height;
                                this.cropX = 0; this.cropY = 0;
                                this.cropWidth  = this.originalWidth;
                                this.cropHeight = this.originalHeight;
                                this.imageElement = img;
                                this.imageData    = re.target.result;
                                this.updateCropData();
                                canvas.setDirty(true);
                            };
                            img.src = re.target.result;
                        };
                        reader.readAsDataURL(file);

                        // Upload
                        const formData = new FormData();
                        formData.append("image", file);
                        formData.append("subfolder", "cropper_uploads");
                        formData.append("type", "input");

                        try {
                            const response = await fetch("/upload/image", { method: "POST", body: formData });
                            if (response.ok) {
                                const res = await response.json();
                                const imagePath = res.subfolder ? `${res.subfolder}/${res.name}` : res.name;
                                this.selectedImagePath = imagePath;
                                if (this.imagePathWidget) this.imagePathWidget.value = imagePath;
                                this.updateCropData();
                            } else {
                                console.error("Upload failed:", await response.text());
                            }
                        } catch (err) {
                            console.error("Upload error:", err);
                        }
                        document.body.removeChild(fileInput);
                    });

                    document.body.appendChild(fileInput);
                    fileInput.click();
                    return true;
                }

                // ── Rotate buttons (active only when image loaded) ──
                if (this.imageElement && this.imageData && this._rotateButtonY > 0) {
                    const rBtnY = this._rotateButtonY;
                    if (localPos[1] >= rBtnY && localPos[1] <= rBtnY + rBtnH) {
                        if (localPos[0] >= margin && localPos[0] <= margin + rBtnW) {
                            // CCW
                            this._onRotate(-1, canvas);
                            return true;
                        }
                        if (localPos[0] >= margin + rBtnW + 8 && localPos[0] <= margin + rectWidth) {
                            // CW
                            this._onRotate(+1, canvas);
                            return true;
                        }
                    }
                }

                // ── Flip buttons (active only when image loaded) ──
                if (this.imageElement && this.imageData && this._flipButtonY > 0) {
                    const fBtnY = this._flipButtonY;
                    if (localPos[1] >= fBtnY && localPos[1] <= fBtnY + rBtnH) {
                        const { w: srcW, h: srcH } = this._rotatedDims();
                        if (localPos[0] >= margin && localPos[0] <= margin + rBtnW) {
                            // Flip H: mirror cropX across the image width
                            this.cropX = srcW - this.cropX - this.cropWidth;
                            this.flipH = !this.flipH;
                            this.updateCropData();
                            canvas.setDirty(true);
                            return true;
                        }
                        if (localPos[0] >= margin + rBtnW + 8 && localPos[0] <= margin + rectWidth) {
                            // Flip V: mirror cropY across the image height
                            this.cropY = srcH - this.cropY - this.cropHeight;
                            this.flipV = !this.flipV;
                            this.updateCropData();
                            canvas.setDirty(true);
                            return true;
                        }
                    }
                }

                // ── Crop handle drag ──
                if (this.imageElement && this.imageData) {
                    const { w: srcW, h: srcH } = this._rotatedDims();

                    if (localPos[1] >= this.imageOffsetY &&
                        localPos[1] <= this.imageOffsetY + this.imageDisplayHeight &&
                        localPos[0] >= this.imageOffsetX &&
                        localPos[0] <= this.imageOffsetX + this.imageDisplayWidth) {

                        const dCX = this.imageOffsetX + this.cropX * this.displayScale;
                        const dCY = this.imageOffsetY + this.cropY * this.displayScale;
                        const dCW = this.cropWidth  * this.displayScale;
                        const dCH = this.cropHeight * this.displayScale;
                        const hs  = 14; // hit size

                        const handles = [
                            { name: 'top-left',     x: dCX,          y: dCY          },
                            { name: 'top-right',    x: dCX + dCW,    y: dCY          },
                            { name: 'bottom-left',  x: dCX,          y: dCY + dCH    },
                            { name: 'bottom-right', x: dCX + dCW,    y: dCY + dCH    },
                            { name: 'top',          x: dCX + dCW/2,  y: dCY          },
                            { name: 'bottom',       x: dCX + dCW/2,  y: dCY + dCH    },
                            { name: 'left',         x: dCX,          y: dCY + dCH/2  },
                            { name: 'right',        x: dCX + dCW,    y: dCY + dCH/2  },
                        ];

                        for (const h of handles) {
                            if (Math.abs(localPos[0] - h.x) < hs && Math.abs(localPos[1] - h.y) < hs) {
                                this.dragging = h.name;
                                this.dragStartX    = localPos[0];
                                this.dragStartY    = localPos[1];
                                this.dragStartCrop = { x: this.cropX, y: this.cropY, w: this.cropWidth, h: this.cropHeight, srcW, srcH };
                                return true;
                            }
                        }

                        // Move whole crop
                        if (localPos[0] >= dCX && localPos[0] <= dCX + dCW &&
                            localPos[1] >= dCY && localPos[1] <= dCY + dCH) {
                            this.dragging = 'move';
                            this.dragStartX    = localPos[0];
                            this.dragStartY    = localPos[1];
                            this.dragStartCrop = { x: this.cropX, y: this.cropY, w: this.cropWidth, h: this.cropHeight, srcW, srcH };
                            return true;
                        }
                    }
                }

                return false;
            };

            nodeType.prototype.onMouseMove = function(e, localPos, canvas) {
                if (!this.dragging || !this.dragStartCrop) return;

                const sc = this.displayScale;
                const dx = (localPos[0] - this.dragStartX) / sc;
                const dy = (localPos[1] - this.dragStartY) / sc;
                const { srcW, srcH } = this.dragStartCrop;
                const min = 10;

                let newX = this.dragStartCrop.x;
                let newY = this.dragStartCrop.y;
                let newW = this.dragStartCrop.w;
                let newH = this.dragStartCrop.h;

                switch (this.dragging) {
                    case 'move':
                        newX = Math.max(0, Math.min(this.dragStartCrop.x + dx, srcW - this.dragStartCrop.w));
                        newY = Math.max(0, Math.min(this.dragStartCrop.y + dy, srcH - this.dragStartCrop.h));
                        break;
                    case 'top':
                        newY = Math.max(0, Math.min(this.dragStartCrop.y + dy, this.dragStartCrop.y + this.dragStartCrop.h - min));
                        newH = this.dragStartCrop.h - (newY - this.dragStartCrop.y);
                        break;
                    case 'bottom':
                        newH = Math.max(min, Math.min(this.dragStartCrop.h + dy, srcH - this.dragStartCrop.y));
                        break;
                    case 'left':
                        newX = Math.max(0, Math.min(this.dragStartCrop.x + dx, this.dragStartCrop.x + this.dragStartCrop.w - min));
                        newW = this.dragStartCrop.w - (newX - this.dragStartCrop.x);
                        break;
                    case 'right':
                        newW = Math.max(min, Math.min(this.dragStartCrop.w + dx, srcW - this.dragStartCrop.x));
                        break;
                    case 'top-left':
                        newY = Math.max(0, Math.min(this.dragStartCrop.y + dy, this.dragStartCrop.y + this.dragStartCrop.h - min));
                        newH = this.dragStartCrop.h - (newY - this.dragStartCrop.y);
                        newX = Math.max(0, Math.min(this.dragStartCrop.x + dx, this.dragStartCrop.x + this.dragStartCrop.w - min));
                        newW = this.dragStartCrop.w - (newX - this.dragStartCrop.x);
                        break;
                    case 'top-right':
                        newY = Math.max(0, Math.min(this.dragStartCrop.y + dy, this.dragStartCrop.y + this.dragStartCrop.h - min));
                        newH = this.dragStartCrop.h - (newY - this.dragStartCrop.y);
                        newW = Math.max(min, Math.min(this.dragStartCrop.w + dx, srcW - this.dragStartCrop.x));
                        break;
                    case 'bottom-left':
                        newH = Math.max(min, Math.min(this.dragStartCrop.h + dy, srcH - this.dragStartCrop.y));
                        newX = Math.max(0, Math.min(this.dragStartCrop.x + dx, this.dragStartCrop.x + this.dragStartCrop.w - min));
                        newW = this.dragStartCrop.w - (newX - this.dragStartCrop.x);
                        break;
                    case 'bottom-right':
                        newW = Math.max(min, Math.min(this.dragStartCrop.w + dx, srcW - this.dragStartCrop.x));
                        newH = Math.max(min, Math.min(this.dragStartCrop.h + dy, srcH - this.dragStartCrop.y));
                        break;
                }

                this.cropX      = Math.max(0, Math.min(Math.round(newX), srcW - min));
                this.cropY      = Math.max(0, Math.min(Math.round(newY), srcH - min));
                this.cropWidth  = Math.max(min, Math.min(Math.round(newW), srcW - this.cropX));
                this.cropHeight = Math.max(min, Math.min(Math.round(newH), srcH - this.cropY));

                this.updateCropData();
                canvas.setDirty(true);
            };

            nodeType.prototype.onMouseUp = function() {
                this.dragging = null;
                this.dragStartCrop = null;
            };

            /**
             * Rotate by 90° in the given direction, transforming the crop rectangle
             * so it stays anchored to the same region of the image.
             * dir: +1 for CW, -1 for CCW
             */
            nodeType.prototype._onRotate = function(dir, canvas) {
                // Capture pre-rotation dims and crop
                const { w: oldW, h: oldH } = this._rotatedDims();
                const ox = this.cropX, oy = this.cropY, ow = this.cropWidth, oh = this.cropHeight;

                // Apply rotation
                this.rotation = (this.rotation + dir * 90 + 360) % 360;
                const { w: newW, h: newH } = this._rotatedDims();

                // Transform crop into the new coordinate space
                let nx, ny, nw, nh;
                if (dir === 1) {
                    // CW 90: point (x,y) maps to (oldH - y - h, x); w and h swap
                    nx = oldH - oy - oh;
                    ny = ox;
                    nw = oh;
                    nh = ow;
                } else {
                    // CCW 90: point (x,y) maps to (y, oldW - x - w); w and h swap
                    nx = oy;
                    ny = oldW - ox - ow;
                    nw = oh;
                    nh = ow;
                }

                // Clamp to new image bounds
                this.cropX      = Math.max(0, Math.min(Math.round(nx), newW - 1));
                this.cropY      = Math.max(0, Math.min(Math.round(ny), newH - 1));
                this.cropWidth  = Math.max(1, Math.min(Math.round(nw), newW - this.cropX));
                this.cropHeight = Math.max(1, Math.min(Math.round(nh), newH - this.cropY));

                this.updateCropData();
                canvas.setDirty(true);
            };

            /** Encode path|x,y,w,h,rotation,flipH,flipV into image_path widget (always serialized) */
            nodeType.prototype.updateCropData = function() {
                if (this.imagePathWidget && this.selectedImagePath) {
                    const crop = `${this.cropX},${this.cropY},${this.cropWidth},${this.cropHeight},${this.rotation},${this.flipH ? 1 : 0},${this.flipV ? 1 : 0}`;
                    this.imagePathWidget.value = `${this.selectedImagePath}|${crop}`;
                }
                if (this.cropDataWidget) {
                    const info = { x: this.cropX, y: this.cropY, width: this.cropWidth, height: this.cropHeight, rotation: this.rotation, flipH: this.flipH, flipV: this.flipV };
                    this.cropDataWidget.value = JSON.stringify(info);
                    this.cropDataWidget.callback?.(this.cropDataWidget.value);
                }
            };
        }
    }
});