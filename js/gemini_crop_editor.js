import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Gemini.ImageCropEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiUploadAndCrop") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Hide the x, y, width, height widgets from user interaction
                for (let w of this.widgets) {
                    if (["x", "y", "width", "height"].includes(w.name)) {
                        w.type = "hidden";
                    }
                }
                
                // Canvas preview properties
                this.image_preview = new Image();
                this.is_drawing = false;
                this.start_x = 0;
                this.start_y = 0;
                this.crop_x = 0;
                this.crop_y = 0;
                this.crop_w = 0;
                this.crop_h = 0;

                const self = this;

                // Redraw canvas whenever the preview image finishes loading
                this.image_preview.onload = function () {
                    self.setDirtyCanvas(true, true);
                };

                // Find the image widget to listen to upload changes
                const imgWidget = this.widgets.find(w => w.name === "image");
                if (imgWidget) {
                    const cb = imgWidget.callback;
                    imgWidget.callback = function(v) {
                        if (cb) cb.apply(this, arguments);
                        if (v) {
                            self.image_preview.src = "/view?filename=" + encodeURIComponent(v) + "&type=input";
                        }
                    };
                    // trigger initial load
                    if (imgWidget.value) {
                        self.image_preview.src = "/view?filename=" + encodeURIComponent(imgWidget.value) + "&type=input";
                    }
                }

                // --- Clipboard paste support ---
                // Use capture phase so we fire BEFORE ComfyUI's own canvas paste handler,
                // then stop propagation so the same image isn't handled twice.
                const pasteHandler = async (e) => {
                    // Only act when this node is selected
                    const selected = app.canvas?.selected_nodes;
                    if (!selected || !selected[self.id]) return;
                    // imgWidget must exist
                    if (!imgWidget) return;

                    const items = e.clipboardData?.items;
                    if (!items) return;

                    let imageItem = null;
                    for (const item of items) {
                        if (item.type.startsWith("image/")) { imageItem = item; break; }
                    }
                    if (!imageItem) return;

                    // Intercept synchronously (must happen before any awaits)
                    e.preventDefault();
                    e.stopPropagation();

                    const blob = imageItem.getAsFile();
                    if (!blob) return;

                    // Unique filename to avoid server-side collisions
                    const ext = (imageItem.type.split("/")[1] || "png").replace(/\+.*$/, "");
                    const formData = new FormData();
                    formData.append("image", blob, `paste_${Date.now()}.${ext}`);
                    formData.append("type", "input");
                    formData.append("overwrite", "false");

                    try {
                        const resp = await fetch("/upload/image", { method: "POST", body: formData });
                        if (!resp.ok) {
                            console.error("[GeminiUploadAndCrop] Upload failed:", resp.status);
                            return;
                        }
                        const data = await resp.json();
                        // ComfyUI returns { name, subfolder, type }
                        const fname = data.name;

                        // Insert into the combo options list (handles both .options.values and .values)
                        const valueList = imgWidget.options?.values ?? imgWidget.values;
                        if (Array.isArray(valueList) && !valueList.includes(fname)) {
                            valueList.push(fname);
                        }

                        imgWidget.value = fname;
                        imgWidget.callback?.call(imgWidget, fname);

                        // Ask ComfyUI to refresh all image-upload combos from the server
                        if (typeof app.refreshComboInNodes === "function") {
                            app.refreshComboInNodes();
                        }

                        self.setDirtyCanvas(true, true);
                    } catch (err) {
                        console.error("[GeminiUploadAndCrop] Paste upload failed:", err);
                    }
                };

                // capture:true — fires before ComfyUI's canvas-level paste handler
                document.addEventListener("paste", pasteHandler, true);

                // Clean up on node removal
                const onRemoved = this.onRemoved;
                this.onRemoved = function () {
                    document.removeEventListener("paste", pasteHandler, true);
                    return onRemoved?.apply(this, arguments);
                };

                return r;
            };
            
            nodeType.prototype.onDrawBackground = function(ctx) {
                if (!this.flags.collapsed) {
                    if (this.image_preview && this.image_preview.complete) {
                        // Calculate where to draw the image on the node
                        const node_width = this.size[0];
                        const img_aspect = this.image_preview.width / this.image_preview.height;
                        
                        const draw_y = 150; // below the standard upload button widgets
                        const max_w = node_width - 20;
                        const draw_w = max_w;
                        const draw_h = draw_w / img_aspect;
                        
                        // Dynamically resize node to fit the image
                        this.size[1] = Math.max(this.size[1], draw_y + draw_h + 30);
                        
                        ctx.drawImage(this.image_preview, 10, draw_y, draw_w, draw_h);
                        
                        // Draw the crop rectangle overlay (Photoshop style)
                        if (this.crop_w > 0 && this.crop_h > 0) {
                            ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
                            ctx.lineWidth = 2;
                            ctx.setLineDash([4, 4]);
                            ctx.strokeRect(10 + this.crop_x, draw_y + this.crop_y, this.crop_w, this.crop_h);
                            
                            // Darken the outside
                            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
                            ctx.beginPath();
                            ctx.rect(10, draw_y, draw_w, draw_h); // Outer
                            ctx.rect(10 + this.crop_x, draw_y + this.crop_y, this.crop_w, this.crop_h); // Inner hole
                            ctx.fill("evenodd");
                            
                            ctx.setLineDash([]);
                        }
                        
                        // Store draw coordinates for mouse interaction mapping
                        this.img_draw_x = 10;
                        this.img_draw_y = draw_y;
                        this.img_draw_w = draw_w;
                        this.img_draw_h = draw_h;
                    }
                }
            };
            
            nodeType.prototype.onMouseDown = function(e, pos) {
                if (this.img_draw_w && pos[0] >= this.img_draw_x && pos[0] <= this.img_draw_x + this.img_draw_w &&
                    pos[1] >= this.img_draw_y && pos[1] <= this.img_draw_y + this.img_draw_h) {
                    this.is_drawing = true;
                    this.start_x = pos[0] - this.img_draw_x;
                    this.start_y = pos[1] - this.img_draw_y;
                    this.crop_x = this.start_x;
                    this.crop_y = this.start_y;
                    this.crop_w = 0;
                    this.crop_h = 0;
                    
                    // Capture mouse to ensure smooth dragging even if cursor leaves node slightly
                    app.canvas.canvas.setPointerCapture(e.pointerId);
                    return true;
                }
                return false;
            };
            
            nodeType.prototype.onMouseMove = function(e, pos) {
                if (this.is_drawing) {
                    const current_x = Math.max(0, Math.min(pos[0] - this.img_draw_x, this.img_draw_w));
                    const current_y = Math.max(0, Math.min(pos[1] - this.img_draw_y, this.img_draw_h));
                    
                    this.crop_x = Math.min(this.start_x, current_x);
                    this.crop_y = Math.min(this.start_y, current_y);
                    this.crop_w = Math.abs(current_x - this.start_x);
                    this.crop_h = Math.abs(current_y - this.start_y);
                    
                    this.setDirtyCanvas(true, true);
                    return true;
                }
                return false;
            };
            
            nodeType.prototype.onMouseUp = function(e, pos) {
                if (this.is_drawing) {
                    this.is_drawing = false;
                    app.canvas.canvas.releasePointerCapture(e.pointerId);
                    
                    // Map display coordinates back to actual image pixel coordinates
                    if (this.image_preview && this.img_draw_w) {
                        const scale_x = this.image_preview.width / this.img_draw_w;
                        const scale_y = this.image_preview.height / this.img_draw_h;
                        
                        const actual_x = Math.round(this.crop_x * scale_x);
                        const actual_y = Math.round(this.crop_y * scale_y);
                        const actual_w = Math.round(this.crop_w * scale_x);
                        const actual_h = Math.round(this.crop_h * scale_y);
                        
                        // Save these mapped coords into the hidden Python node widgets
                        for (let w of this.widgets) {
                            if (w.name === "x") w.value = actual_x;
                            if (w.name === "y") w.value = actual_y;
                            if (w.name === "width") w.value = actual_w;
                            if (w.name === "height") w.value = actual_h;
                        }
                    }
                    return true;
                }
                return false;
            };
        }
    }
});
