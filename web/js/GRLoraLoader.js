import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "GRLoraLoaderExtension",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GRLoraLoader") {
            // console.log("=== Registering GRLoraLoader extension ===");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                // console.log("=== GRLoraLoader onNodeCreated START ===");
                
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.loraWidgets = [];
                this.loraCounter = 0;
                this.numLorasToEnable = 1;
                
                // Set minimum width for the node
                this.size[0] = Math.max(this.size[0] || 0, 300);
                
                // Try to load saved configuration
                this.loadConfiguration();
                
                // ROW 1: Add LoRA button
                const addLoraContainer = document.createElement("div");
                addLoraContainer.style.display = "flex";
                addLoraContainer.style.gap = "10px";
                addLoraContainer.style.padding = "0px 5px 2px 5px";
                addLoraContainer.style.width = "100%";
                addLoraContainer.style.boxSizing = "border-box";
                
                const addLoraBtn = document.createElement("button");
                addLoraBtn.textContent = "Add LoRA";
                addLoraBtn.style.flex = "1";
                addLoraBtn.style.height = "20px";
                addLoraBtn.style.padding = "0";
                addLoraBtn.style.cursor = "pointer";
                addLoraBtn.style.fontSize = "12px";
                addLoraBtn.onclick = () => {
                    // console.log("Add LoRA button clicked");
                    this.showLoraChooser();
                };
                
                addLoraContainer.appendChild(addLoraBtn);
                
                const addLoraWidget = this.addDOMWidget("add_lora_button", "div", addLoraContainer, {
                    getValue: () => null,
                    setValue: () => {},
                });
                addLoraWidget.computeSize = () => [0, 28];
                
                // ROW 2: Enable input and Randomize LoRAs button
                const randomizeContainer = document.createElement("div");
                randomizeContainer.style.display = "flex";
                randomizeContainer.style.gap = "10px";
                randomizeContainer.style.padding = "0px 5px 2px 5px";
                randomizeContainer.style.width = "100%";
                randomizeContainer.style.boxSizing = "border-box";
 
                
                const label = document.createElement("span");
                label.textContent = "Enable:";
                label.style.color = "#ffffff";
                label.style.fontSize = "12px";
                label.style.whiteSpace = "nowrap";
                
                const numLorasInput = document.createElement("input");
                numLorasInput.type = "number";
                numLorasInput.value = "1";
                numLorasInput.min = "0";
                numLorasInput.step = "1";
                numLorasInput.style.width = "60px";
                numLorasInput.style.height = "20px";
                numLorasInput.style.padding = "2px 5px";
                numLorasInput.style.backgroundColor = "#1a1a1a";
                numLorasInput.style.color = "#ffffff";
                numLorasInput.style.border = "1px solid #444";
                numLorasInput.style.borderRadius = "3px";
                numLorasInput.style.fontSize = "12px";
                numLorasInput.onchange = () => {
                    this.numLorasToEnable = parseInt(numLorasInput.value) || 1;
                    // console.log("Number of LoRAs to enable set to:", this.numLorasToEnable);
                };
                
                const randomizeBtn = document.createElement("button");
                randomizeBtn.textContent = "🎲 Randomize LoRAs";
                randomizeBtn.style.flex = "1";
                randomizeBtn.style.height = "20px";
                randomizeBtn.style.padding = "0";
                randomizeBtn.style.cursor = "pointer";
                randomizeBtn.style.fontSize = "12px";
                randomizeBtn.onclick = () => {
                    // console.log("Randomize LoRAs button clicked");
                    this.randomizeLoras();
                };
                
                randomizeContainer.appendChild(label);
                randomizeContainer.appendChild(numLorasInput);
                randomizeContainer.appendChild(randomizeBtn);
                
                const randomizeWidget = this.addDOMWidget("randomize_controls", "div", randomizeContainer, {
                    getValue: () => this.numLorasToEnable,
                    setValue: (v) => {
                        this.numLorasToEnable = v || 1;
                        numLorasInput.value = this.numLorasToEnable;
                    },
                });
                randomizeWidget.computeSize = () => [0, 28];
                
                this.numLorasInput = numLorasInput;
                
                // ROW 3: Randomize Enabled and Normalize Enabled
                const enabledControlsContainer = document.createElement("div");
                enabledControlsContainer.style.display = "flex";
                enabledControlsContainer.style.gap = "10px";
                enabledControlsContainer.style.padding = "0px 5px 2px 5px";
                enabledControlsContainer.style.width = "100%";
                enabledControlsContainer.style.boxSizing = "border-box";
                
                const randomizeEnabledBtn = document.createElement("button");
                randomizeEnabledBtn.textContent = "🎲 Randomize Enabled";
                randomizeEnabledBtn.style.flex = "1";
                randomizeEnabledBtn.style.height = "20px";
                randomizeEnabledBtn.style.padding = "0";
                randomizeEnabledBtn.style.cursor = "pointer";
                randomizeEnabledBtn.style.fontSize = "12px";
                randomizeEnabledBtn.onclick = () => {
                    // console.log("Randomize Enabled button clicked");
                    this.randomizeEnabledWeights();
                };
                
                const normalizeEnabledBtn = document.createElement("button");
                normalizeEnabledBtn.textContent = "⚖️ Normalize Enabled";
                normalizeEnabledBtn.style.flex = "1";
                normalizeEnabledBtn.style.height = "20px";
                normalizeEnabledBtn.style.padding = "0";
                normalizeEnabledBtn.style.cursor = "pointer";
                normalizeEnabledBtn.style.fontSize = "12px";
                normalizeEnabledBtn.onclick = () => {
                    // console.log("Normalize Enabled button clicked");
                    this.normalizeEnabledWeights();
                };
                
                enabledControlsContainer.appendChild(randomizeEnabledBtn);
                enabledControlsContainer.appendChild(normalizeEnabledBtn);
                
                const enabledControlsWidget = this.addDOMWidget("enabled_controls", "div", enabledControlsContainer, {
                    getValue: () => null,
                    setValue: () => {},
                });
                enabledControlsWidget.computeSize = () => [0, 28];
                
                this.setSize([Math.max(200, this.size[0]), Math.max(140, this.size[1])]);
                this.setDirtyCanvas(true, true);
                
                // console.log("=== GRLoraLoader onNodeCreated END ===");
                return r;
            };
            
            // Override onResize to save when user manually resizes
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }
                // Save configuration when user resizes
                if (this.id) {
                    this.saveConfiguration();
                }
            };
            
            // Override onMoved to save position when user moves the node
            nodeType.prototype.onMoved = function() {
                // Save configuration when user moves the node
                if (this.id) {
                    this.saveConfiguration();
                }
            };
            
            // Method to save configuration to file
            nodeType.prototype.saveConfiguration = async function() {
                if (!this.id) {
                    console.warn("Node ID not available yet, skipping save");
                    return;
                }
                
                const loraWidgetData = this.loraWidgets
                    .filter(w => w && typeof w.getValue === 'function')
                    .map(w => w.getValue());
                
                const config = {
                    num_loras_to_enable: this.numLorasToEnable,
                    lora_widgets: loraWidgetData,
                    node_size: this.size,
                    node_pos: this.pos
                };
                
                // console.log("=== SAVING CONFIGURATION ===");
                // console.log("Node ID:", this.id);
                // console.log("Config:", JSON.stringify(config, null, 2));
                
                try {
                    const response = await api.fetchApi("/gr_lora_loader/save_config", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            node_id: this.id,
                            config: config
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // console.log("✓ Configuration saved successfully:", result);
                    } else {
                        console.error("✗ Failed to save configuration:", result);
                    }
                } catch (error) {
                    console.error("✗ Error saving configuration:", error);
                }
            };
            
            // Method to load configuration from file
            nodeType.prototype.loadConfiguration = async function() {
                if (!this.id) {
                    console.warn("Node ID not available yet, skipping load");
                    return;
                }
                
                // console.log("=== LOADING CONFIGURATION ===");
                // console.log("Node ID:", this.id);
                
                try {
                    const response = await api.fetchApi(`/gr_lora_loader/load_config?node_id=${this.id}`);
                    
                    if (response.ok) {
                        const data = await response.json();
                        // console.log("Received data:", data);
                        
                        if (data.config && data.config.lora_widgets) {
                            // console.log("Found saved configuration with", data.config.lora_widgets.length, "LoRAs");
                            
                            if (data.config.num_loras_to_enable !== undefined) {
                                this.numLorasToEnable = data.config.num_loras_to_enable;
                                if (this.numLorasInput) {
                                    this.numLorasInput.value = this.numLorasToEnable;
                                }
                            }
                            
                            // Restore node size and position
                            if (data.config.node_size) {
                                this.size = [...data.config.node_size];
                                // console.log("Restored node size:", this.size);
                            }
                            if (data.config.node_pos) {
                                this.pos = [...data.config.node_pos];
                                // console.log("Restored node position:", this.pos);
                            }
                            
                            setTimeout(() => {
                                for (const loraData of data.config.lora_widgets) {
                                    if (loraData && loraData.lora) {
                                        // console.log("Restoring LoRA:", loraData.lora);
                                        this.addLoraWidget(loraData.lora, true);
                                        const lastWidget = this.loraWidgets[this.loraWidgets.length - 1];
                                        if (lastWidget) {
                                            const checkbox = lastWidget.loraContainer?.querySelector('input[type="checkbox"]');
                                            const strengthInput = lastWidget.loraContainer?.querySelector('input[type="number"]');
                                            
                                            if (checkbox) {
                                                checkbox.checked = loraData.on !== false;
                                            }
                                            if (strengthInput) {
                                                strengthInput.value = loraData.strength || 1.0;
                                            }
                                            
                                            console.log(`✓ Restored LoRA: ${loraData.lora}, enabled: ${loraData.on}, strength: ${loraData.strength}`);
                                        }
                                    }
                                }
                                // Don't auto-resize after loading - keep saved size
                                if (!data.config.node_size) {
                                    this.setSize([Math.max(300, this.size[0]), this.computeSize()[1]]);
                                }
                                this.setDirtyCanvas(true, true);
                            }, 100);
                        } else {
                            // console.log("No saved configuration found");
                        }
                    }
                } catch (error) {
                    console.error("✗ Error loading configuration:", error);
                }
            };
            
            // Method to show LoRA chooser
            nodeType.prototype.showLoraChooser = async function() {
                try {
                    const response = await api.fetchApi("/gr_lora_loader/loras");
                    
                    if (response.ok) {
                        const loras = await response.json();
                        // console.log("Available LoRAs:", loras);
                        
                        this.showLoraSearchDialog(loras);
                    } else {
                        console.error("Failed to fetch LoRAs");
                        alert("Failed to fetch LoRAs from server");
                    }
                } catch (error) {
                    console.error("Error fetching LoRAs:", error);
                    alert("Error fetching LoRAs: " + error.message);
                }
            };
            
            // Method to show searchable LoRA dialog
            nodeType.prototype.showLoraSearchDialog = function(loras) {
                const node = this;
                
                const overlay = document.createElement("div");
                overlay.style.position = "fixed";
                overlay.style.top = "0";
                overlay.style.left = "0";
                overlay.style.width = "100%";
                overlay.style.height = "100%";
                overlay.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
                overlay.style.zIndex = "10000";
                overlay.style.display = "flex";
                overlay.style.justifyContent = "center";
                overlay.style.alignItems = "center";
                
                const dialog = document.createElement("div");
                dialog.style.backgroundColor = "#2a2a2a";
                dialog.style.border = "2px solid #555";
                dialog.style.borderRadius = "8px";
                dialog.style.padding = "10px";
                dialog.style.width = "720px";
                dialog.style.height = "700px";
                dialog.style.maxHeight = "80vh";
                dialog.style.display = "flex";
                dialog.style.flexDirection = "column";
                dialog.style.gap = "10px";
                
                const title = document.createElement("h3");
                title.textContent = "Select LoRA";
                title.style.color = "#ffffff";
                title.style.margin = "0 0 10px 0";
                
                const searchInput = document.createElement("input");
                searchInput.type = "text";
                searchInput.placeholder = "Type to search...";
                searchInput.style.width = "100%";
                searchInput.style.padding = "10px";
                searchInput.style.backgroundColor = "#1a1a1a";
                searchInput.style.color = "#ffffff";
                searchInput.style.border = "1px solid #444";
                searchInput.style.borderRadius = "4px";
                searchInput.style.fontSize = "12px";
                searchInput.style.boxSizing = "border-box";
                
                const resultsContainer = document.createElement("div");
                resultsContainer.style.maxHeight = "680px";
                resultsContainer.style.maxWidth = "700px";
                resultsContainer.style.overflowY = "auto";
                resultsContainer.style.border = "1px solid #444";
                resultsContainer.style.borderRadius = "4px";
                resultsContainer.style.backgroundColor = "#1a1a1a";
                
                const renderResults = (filteredLoras) => {
                    resultsContainer.innerHTML = "";
                    
                    if (filteredLoras.length === 0) {
                        const noResults = document.createElement("div");
                        noResults.textContent = "No LoRAs found";
                        noResults.style.color = "#888";
                        noResults.style.padding = "20px";
                        noResults.style.textAlign = "center";
                        resultsContainer.appendChild(noResults);
                        return;
                    }
                    
                    filteredLoras.forEach(lora => {
                        const item = document.createElement("div");
                        item.textContent = lora;
                        item.style.padding = "10px";
                        item.style.color = "#ffffff";
                        item.style.cursor = "pointer";
                        item.style.borderBottom = "1px solid #333";
                        
                        item.onmouseover = () => {
                            item.style.backgroundColor = "#3a3a3a";
                        };
                        
                        item.onmouseout = () => {
                            item.style.backgroundColor = "transparent";
                        };
                        
                        item.onclick = () => {
                            // console.log("Selected LoRA:", lora);
                            node.addLoraWidget(lora);
                            document.body.removeChild(overlay);
                        };
                        
                        resultsContainer.appendChild(item);
                    });
                };
                
                renderResults(loras);
                
                searchInput.oninput = () => {
                    const searchTerm = searchInput.value.toLowerCase();
                    const filtered = loras.filter(lora => 
                        lora.toLowerCase().includes(searchTerm)
                    );
                    renderResults(filtered);
                };
                
                const closeButton = document.createElement("button");
                closeButton.textContent = "Cancel";
                closeButton.style.padding = "10px";
                closeButton.style.backgroundColor = "#444";
                closeButton.style.color = "#ffffff";
                closeButton.style.border = "none";
                closeButton.style.borderRadius = "4px";
                closeButton.style.cursor = "pointer";
                closeButton.style.fontSize = "14px";
                closeButton.onclick = () => {
                    document.body.removeChild(overlay);
                };
                
                overlay.onclick = (e) => {
                    if (e.target === overlay) {
                        document.body.removeChild(overlay);
                    }
                };
                
                const handleEscape = (e) => {
                    if (e.key === "Escape") {
                        document.body.removeChild(overlay);
                        document.removeEventListener("keydown", handleEscape);
                    }
                };
                document.addEventListener("keydown", handleEscape);
                
                dialog.appendChild(title);
                dialog.appendChild(searchInput);
                dialog.appendChild(resultsContainer);
                dialog.appendChild(closeButton);
                overlay.appendChild(dialog);
                document.body.appendChild(overlay);
                
                setTimeout(() => searchInput.focus(), 100);
            };
            
            // Method to randomize LoRAs
            nodeType.prototype.randomizeLoras = function() {
                const numToEnable = this.numLorasToEnable || 1;
                
                if (this.loraWidgets.length === 0) {
                    alert("No LoRAs added yet. Add some LoRAs first.");
                    return;
                }
                
                console.log(`Randomizing: enabling ${numToEnable} out of ${this.loraWidgets.length} LoRAs`);
                
                const indices = this.loraWidgets.map((_, i) => i);
                
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                
                this.loraWidgets.forEach((widget, idx) => {
                    const shouldEnable = indices.indexOf(idx) < numToEnable;
                    
                    const checkbox = widget.loraContainer?.querySelector('input[type="checkbox"]');
                    if (checkbox) {
                        checkbox.checked = shouldEnable;
                        console.log(`LoRA ${idx}: ${shouldEnable ? 'enabled' : 'disabled'}`);
                    }
                });
                
                this.setDirtyCanvas(true, true);
                // console.log("LoRAs randomized successfully");
                this.randomizeEnabledWeights()
                // console.log("Randomize LoRAs complete, auto-saving");
//                this.saveConfiguration();
            };
            
            // Method to randomize enabled LoRA weights (sum to 1.0)
            nodeType.prototype.randomizeEnabledWeights = function() {
                if (this.loraWidgets.length === 0) {
                    alert("No LoRAs added yet. Add some LoRAs first.");
                    return;
                }
                
                const enabledWidgets = this.loraWidgets.filter(widget => {
                    const checkbox = widget.loraContainer?.querySelector('input[type="checkbox"]');
                    return checkbox && checkbox.checked;
                });
                
                if (enabledWidgets.length === 0) {
                    alert("No enabled LoRAs to randomize. Enable some LoRAs first.");
                    return;
                }
                
                console.log(`Randomizing weights for ${enabledWidgets.length} enabled LoRAs`);
                
                let randomWeights = enabledWidgets.map(() => Math.random());
                
                const sum = randomWeights.reduce((a, b) => a + b, 0);
                randomWeights = randomWeights.map(w => w / sum);
                
                enabledWidgets.forEach((widget, idx) => {
                    const strengthInput = widget.loraContainer?.querySelector('input[type="number"]');
                    if (strengthInput) {
                        const weight = Math.round(randomWeights[idx] * 100) / 100;
                        strengthInput.value = weight.toFixed(2);
                        console.log(`LoRA ${idx}: weight set to ${weight}`);
                    }
                });
                
                this.setDirtyCanvas(true, true);
                // console.log("Enabled LoRA weights randomized successfully (sum = 1.0)");
                
                // console.log("Randomize Enabled complete, auto-saving");
                this.saveConfiguration();
            };
            
            // Method to normalize enabled LoRA weights (equal distribution)
            nodeType.prototype.normalizeEnabledWeights = function() {
                if (this.loraWidgets.length === 0) {
                    alert("No LoRAs added yet. Add some LoRAs first.");
                    return;
                }
                
                const enabledWidgets = this.loraWidgets.filter(widget => {
                    const checkbox = widget.loraContainer?.querySelector('input[type="checkbox"]');
                    return checkbox && checkbox.checked;
                });
                
                if (enabledWidgets.length === 0) {
                    alert("No enabled LoRAs to normalize. Enable some LoRAs first.");
                    return;
                }
                
                console.log(`Normalizing weights for ${enabledWidgets.length} enabled LoRAs`);
                
                const equalWeight = 1.0 / enabledWidgets.length;
                
                enabledWidgets.forEach((widget, idx) => {
                    const strengthInput = widget.loraContainer?.querySelector('input[type="number"]');
                    if (strengthInput) {
                        strengthInput.value = equalWeight.toFixed(4);
                        console.log(`LoRA ${idx}: weight set to ${equalWeight.toFixed(4)}`);
                    }
                });
                
                this.setDirtyCanvas(true, true);
                console.log(`Enabled LoRA weights normalized to ${equalWeight.toFixed(4)} each`);
                
                // console.log("Normalize Enabled complete, auto-saving");
                this.saveConfiguration();
            };
            
            // Method to add a new LoRA widget
            nodeType.prototype.addLoraWidget = function(loraName, skipSave = false) {
                this.loraCounter++;
                const widgetName = `lora_${this.loraCounter}`;
                
                // console.log("Adding LoRA widget:", widgetName, "with lora:", loraName);
                
                const loraContainer = document.createElement("div");
                loraContainer.style.display = "flex";
                loraContainer.style.gap = "2px";
                loraContainer.style.padding = "4px 5px";
                loraContainer.style.alignItems = "center";
                loraContainer.style.backgroundColor = "#2a2a2a";
                loraContainer.style.marginBottom = "2px";
                loraContainer.style.borderRadius = "3px";
                
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = true;
                checkbox.style.width = "14px";
                checkbox.style.height = "14px";
                checkbox.style.cursor = "pointer";
                checkbox.onchange = () => {
                    // console.log("Checkbox changed, auto-saving");
                    this.saveConfiguration();
                };
                
                const nameLabel = document.createElement("span");
                nameLabel.textContent = loraName;
                nameLabel.style.flex = "1";
                nameLabel.style.color = "#ffffff";
                nameLabel.style.fontSize = "12px";
                nameLabel.style.overflow = "hidden";
                nameLabel.style.textOverflow = "ellipsis";
                nameLabel.style.whiteSpace = "nowrap";
                
                const strengthInput = document.createElement("input");
                strengthInput.type = "number";
                strengthInput.value = "1.0";
                strengthInput.step = "0.05";
                strengthInput.min = "0";
                strengthInput.max = "5";
                strengthInput.style.width = "50px";
                strengthInput.style.height = "18px";
                strengthInput.style.padding = "2px 5px";
                strengthInput.style.backgroundColor = "#1a1a1a";
                strengthInput.style.color = "#ffffff";
                strengthInput.style.border = "1px solid #444";
                strengthInput.style.borderRadius = "3px";
                strengthInput.style.fontSize = "12px";
                strengthInput.onchange = () => {
                    // console.log("Strength changed, auto-saving");
                    this.saveConfiguration();
                };
                strengthInput.oninput = () => {
                    // console.log("Strength input changed, auto-saving");
                    this.saveConfiguration();
                };
                
                const removeBtn = document.createElement("button");
                removeBtn.textContent = "×";
                removeBtn.style.width = "16px";
                removeBtn.style.height = "16px";
                removeBtn.style.padding = "0";
                removeBtn.style.cursor = "pointer";
                removeBtn.style.fontSize = "12px";
                removeBtn.style.backgroundColor = "#c44";
                removeBtn.style.color = "#fff";
                removeBtn.style.border = "none";
                removeBtn.style.borderRadius = "3px";
                removeBtn.onclick = () => {
                    // console.log("Removing LoRA widget:", widgetName);
                    const widgetIndex = this.widgets.findIndex(w => w.loraContainer === loraContainer);
                    if (widgetIndex !== -1) {
                        this.widgets.splice(widgetIndex, 1);
                    }
                    const loraIndex = this.loraWidgets.findIndex(w => w.loraContainer === loraContainer);
                    if (loraIndex !== -1) {
                        this.loraWidgets.splice(loraIndex, 1);
                    }
                    loraContainer.remove();
//                    this.setSize([Math.max(300, this.size[0]), this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                    
                    // console.log("LoRA removed, auto-saving");
                    this.saveConfiguration();
                };
                
                loraContainer.appendChild(checkbox);
                loraContainer.appendChild(nameLabel);
                loraContainer.appendChild(strengthInput);
                loraContainer.appendChild(removeBtn);
                
                const widgetData = {
                    type: "lora_widget",
                    name: widgetName,
                    loraContainer: loraContainer,
                    loraName: loraName,
                    checkbox: checkbox,
                    strengthInput: strengthInput,
                    getValue: function() {
                        return {
                            on: checkbox.checked,
                            lora: loraName,
                            strength: parseFloat(strengthInput.value) || 1.0,
                            strengthTwo: parseFloat(strengthInput.value) || 1.0
                        };
                    },
                    setValue: function(value) {
                        if (value && typeof value === "object") {
                            checkbox.checked = value.on !== false;
                            strengthInput.value = value.strength || 1.0;
                        }
                    }
                };
                
                const htmlWidget = this.addDOMWidget(widgetName, "div", loraContainer, {
                    getValue: () => widgetData.getValue(),
                    setValue: (v) => widgetData.setValue(v),
                });
                htmlWidget.computeSize = () => [0, 20];
                htmlWidget.loraContainer = loraContainer;
                
                htmlWidget.getValue = widgetData.getValue;
                htmlWidget.setValue = widgetData.setValue;
                
                this.loraWidgets.push(htmlWidget);
                
//                const currentSize = this.computeSize();
//                this.setSize([Math.max(300, this.size[0]), currentSize[1]]);
                this.setDirtyCanvas(true, true);
                
                if (!skipSave) {
                    // console.log("Auto-saving after adding LoRA");
                    this.saveConfiguration();
                }
            };
            
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
            };
            
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }
                
                o.lora_widgets = this.loraWidgets
                    .filter(w => w && typeof w.getValue === 'function')
                    .map(w => w.getValue());
                o.num_loras_to_enable = this.numLorasToEnable;
                
                if (!o.widgets_values) {
                    o.widgets_values = [];
                }
                this.loraWidgets.forEach(w => {
                    if (w && typeof w.getValue === 'function') {
                        o.widgets_values.push(w.getValue());
                    }
                });
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                // console.log("=== Configuring node, data:", o);
                
                if (this.loraWidgets && this.loraWidgets.length > 0) {
                    // console.log("Clearing existing lora widgets");
                    const existingLoraWidgets = [...this.loraWidgets];
                    existingLoraWidgets.forEach(widget => {
                        const idx = this.widgets.indexOf(widget);
                        if (idx !== -1) {
                            this.widgets.splice(idx, 1);
                        }
                    });
                    this.loraWidgets = [];
                }
                
                if (o.num_loras_to_enable !== undefined) {
                    this.numLorasToEnable = o.num_loras_to_enable;
                    if (this.numLorasInput) {
                        this.numLorasInput.value = this.numLorasToEnable;
                    }
                }
                
                if (o.lora_widgets && Array.isArray(o.lora_widgets)) {
                    // console.log("Restoring lora widgets:", o.lora_widgets);
                    for (const loraData of o.lora_widgets) {
                        if (loraData && loraData.lora) {
                            this.addLoraWidget(loraData.lora, true);
                            const lastWidget = this.loraWidgets[this.loraWidgets.length - 1];
                            if (lastWidget) {
                                const checkbox = lastWidget.loraContainer?.querySelector('input[type="checkbox"]');
                                const strengthInput = lastWidget.loraContainer?.querySelector('input[type="number"]');
                                
                                if (checkbox) {
                                    checkbox.checked = loraData.on !== false;
                                }
                                if (strengthInput) {
                                    strengthInput.value = loraData.strength || 1.0;
                                }
                                
                                console.log(`Restored LoRA: ${loraData.lora}, enabled: ${loraData.on}, strength: ${loraData.strength}`);
                            }
                        }
                    }
                }
                else if (o.widgets_values && Array.isArray(o.widgets_values)) {
                    // console.log("Restoring from widgets_values:", o.widgets_values);
                    for (const value of o.widgets_values) {
                        if (value && typeof value === "object" && value.lora) {
                            this.addLoraWidget(value.lora, true);
                            const lastWidget = this.loraWidgets[this.loraWidgets.length - 1];
                            if (lastWidget) {
                                const checkbox = lastWidget.loraContainer?.querySelector('input[type="checkbox"]');
                                const strengthInput = lastWidget.loraContainer?.querySelector('input[type="number"]');
                                
                                if (checkbox) {
                                    checkbox.checked = value.on !== false;
                                }
                                if (strengthInput) {
                                    strengthInput.value = value.strength || 1.0;
                                }
                                
                                console.log(`Restored LoRA: ${value.lora}, enabled: ${value.on}, strength: ${value.strength}`);
                            }
                        }
                    }
                }
                
                // console.log("Final lora widgets count:", this.loraWidgets.length);
                
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
            };
        }
    }
});
