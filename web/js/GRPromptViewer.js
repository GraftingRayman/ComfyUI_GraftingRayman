import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "GRPromptViewerExtension",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GRPromptViewer") {
            console.log("=== Registering GRPromptViewer extension ===");

            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            const originalOnExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onNodeCreated = function() {
                console.log("=== GRPromptViewer onNodeCreated START ===");
                
                const r = originalNodeCreated ? originalNodeCreated.apply(this, arguments) : undefined;
                
                this.contentWidget = this.widgets.find(w => w.name === "content");
                this.editedWidget = this.widgets.find(w => w.name === "edited");
                
                // Hide the content widget
                if (this.contentWidget) {
                    this.contentWidget.computeSize = () => [0, -4];
                    if (this.contentWidget.inputEl) {
                        this.contentWidget.inputEl.style.display = "none";
                    }
                }
                
                // Hide the edited widget
                if (this.editedWidget) {
                    this.editedWidget.computeSize = () => [0, -15];
                    if (this.editedWidget.inputEl) {
                        this.editedWidget.inputEl.style.display = "none";
                    }
                    this.editedWidget.value = false;
                }
                
                // Create preview widget
                const previewWidget = ComfyWidgets["STRING"](
                    this, 
                    "content_preview", 
                    ["STRING", { multiline: true }], 
                    app
                )?.widget;
                
                if (previewWidget) {
                    previewWidget.inputEl.readOnly = false;
                    previewWidget.inputEl.style.opacity = 1.0;
                    previewWidget.inputEl.style.fontFamily = "monospace";
                    previewWidget.inputEl.style.fontSize = "11px";
                    previewWidget.inputEl.rows = 15;
                    previewWidget.value = "Select a folder and file to view contents...";
                    
                    this.contentPreview = previewWidget;
                    this.originalContent = previewWidget.value;
                    this.isModified = false;
                    this.currentFolder = "";
                    this.currentFile = "";
                    this.hasAssociatedImage = false;
                    this.hasCaptionFile = false;
                    
                    previewWidget.inputEl.addEventListener('input', () => {
                        const isNowModified = (previewWidget.value !== this.originalContent);
                        
                        if (this.isModified !== isNowModified) {
                            this.isModified = isNowModified;
                        }
                        
                        if (this.saveButton) {
                            this.saveButton.disabled = !this.isModified;
                        }
                        
                        if (this.editedWidget) {
                            this.editedWidget.value = Boolean(this.isModified);
                            if (this.editedWidget.callback) {
                                this.editedWidget.callback(this.editedWidget.value);
                            }
                        }
                        
                        if (this.contentWidget) {
                            this.contentWidget.value = previewWidget.value;
                        }
                        
                        this.setDirtyCanvas(true, true);
                    });
                }
                
                // Create button container
                const buttonContainer = document.createElement("div");
                buttonContainer.style.display = "flex";
                buttonContainer.style.gap = "10px";
                buttonContainer.style.padding = "10px";
                buttonContainer.style.width = "100%";
                buttonContainer.style.boxSizing = "border-box";
                
                const saveBtn = document.createElement("button");
                saveBtn.textContent = "Save";
                saveBtn.style.flex = "1";
                saveBtn.style.height = "35px";
                saveBtn.style.cursor = "pointer";
                saveBtn.disabled = true;
                saveBtn.onclick = () => this.saveFile(false);
                
                const saveAsBtn = document.createElement("button");
                saveAsBtn.textContent = "Save As";
                saveAsBtn.style.flex = "1";
                saveAsBtn.style.height = "35px";
                saveAsBtn.style.cursor = "pointer";
                saveAsBtn.onclick = () => this.saveFile(true);
                
                const clearBtn = document.createElement("button");
                clearBtn.textContent = "Clear";
                clearBtn.style.flex = "1";
                clearBtn.style.height = "35px";
                clearBtn.style.cursor = "pointer";
                clearBtn.onclick = () => this.clearContent();
                
                buttonContainer.appendChild(saveBtn);
                buttonContainer.appendChild(saveAsBtn);
                buttonContainer.appendChild(clearBtn);
                
                const htmlWidget = this.addDOMWidget("buttons", "div", buttonContainer, {
                    getValue: () => null,
                    setValue: () => {},
                });
                htmlWidget.computeSize = () => [0, 55];
                
                this.saveButton = saveBtn;
                this.saveAsButton = saveAsBtn;
                this.clearButton = clearBtn;
                
                // Create caption generation container
                const captionContainer = document.createElement("div");
                captionContainer.style.display = "none";
                captionContainer.style.padding = "10px";
                captionContainer.style.backgroundColor = "#2a2a2a";
                captionContainer.style.borderRadius = "4px";
                captionContainer.style.marginBottom = "10px";
                
                const captionMessage = document.createElement("div");
                captionMessage.textContent = "Image found without caption. Generate caption?";
                captionMessage.style.marginBottom = "10px";
                captionMessage.style.color = "#ffcc00";
                captionMessage.style.fontSize = "12px";
                
                const captionButtonContainer = document.createElement("div");
                captionButtonContainer.style.display = "flex";
                captionButtonContainer.style.gap = "10px";
                
                const generateBtn = document.createElement("button");
                generateBtn.textContent = "Yes - Generate Caption";
                generateBtn.style.flex = "1";
                generateBtn.style.height = "30px";
                generateBtn.style.cursor = "pointer";
                generateBtn.style.fontSize = "12px";
                generateBtn.style.backgroundColor = "#4a9eff";
                generateBtn.style.color = "white";
                generateBtn.style.border = "none";
                generateBtn.style.borderRadius = "4px";
                generateBtn.onclick = () => this.generateCaption();
                
                const skipBtn = document.createElement("button");
                skipBtn.textContent = "No - Skip";
                skipBtn.style.flex = "1";
                skipBtn.style.height = "30px";
                skipBtn.style.cursor = "pointer";
                skipBtn.style.fontSize = "12px";
                skipBtn.onclick = () => {
                    captionContainer.style.display = "none";
                    if (this.captionWidget) {
                        this.captionWidget.computeSize = () => [0, 0];
                    }
                };
                
                captionButtonContainer.appendChild(generateBtn);
                captionButtonContainer.appendChild(skipBtn);
                captionContainer.appendChild(captionMessage);
                captionContainer.appendChild(captionButtonContainer);
                
                this.captionWidget = this.addDOMWidget("caption_gen", "div", captionContainer, {
                    getValue: () => null,
                    setValue: () => {},
                });
                this.captionWidget.computeSize = () => [0, 0];
                
                this.captionContainer = captionContainer;
                this.generateCaptionButton = generateBtn;
                
                // Helper functions
                this.isInputConnected = function(inputName) {
                    if (!this.inputs) return false;
                    const input = this.inputs.find(i => i.name === inputName);
                    return input && input.link != null;
                };
                
                const originalGetInputData = this.getInputData;
                this.getInputData = function(slot) {
                    const input = this.inputs?.[slot];
                    if (input && input.name === "content") {
                        if (input.link != null) {
                            return originalGetInputData ? originalGetInputData.call(this, slot) : undefined;
                        } else if (this.contentPreview) {
                            return this.contentPreview.value;
                        }
                    }
                    return originalGetInputData ? originalGetInputData.call(this, slot) : undefined;
                };
                
                const hideCaptionGenerator = () => {
                    if (this.captionContainer) {
                        this.captionContainer.style.display = "none";
                    }
                    if (this.captionWidget) {
                        this.captionWidget.computeSize = () => [0, 0];
                    }
                };
                
                const showCaptionGenerator = () => {
                    if (this.captionContainer) {
                        this.captionContainer.style.display = "block";
                    }
                    if (this.captionWidget) {
                        this.captionWidget.computeSize = () => [0, 80];
                    }
                };
                
                const clearImageDisplay = () => {
                    if (this.imageDisplayContainer) {
                        this.imageDisplayContainer.style.display = "none";
                        this.imageDisplayContainer.innerHTML = "";
                    }
                    if (this.imageWidget) {
                        this.imageWidget.computeSize = () => [0, 0];
                    }
                };
                
                const displayImage = (imageDataUrl) => {
                    if (!this.imageDisplayContainer) {
                        this.imageDisplayContainer = document.createElement("div");
                        this.imageDisplayContainer.style.padding = "10px";
                        this.imageDisplayContainer.style.textAlign = "center";
                        this.imageDisplayContainer.style.borderTop = "1px solid #444";
                        this.imageDisplayContainer.style.marginTop = "10px";
                        
                        this.imageWidget = this.addDOMWidget("image_preview", "div", this.imageDisplayContainer, {
                            getValue: () => null,
                            setValue: () => {},
                        });
                    }
                    
                    this.imageDisplayContainer.style.display = "block";
                    this.imageDisplayContainer.innerHTML = "";
                    
                    const img = document.createElement("img");
                    img.src = imageDataUrl;
                    img.style.maxWidth = "100%";
                    img.style.maxHeight = "400px";
                    img.style.borderRadius = "4px";
                    img.style.boxShadow = "0 2px 8px rgba(0,0,0,0.3)";
                    
                    this.imageDisplayContainer.appendChild(img);
                    this.imageWidget.computeSize = () => [0, 420];
                };
                
                const loadAssociatedImage = async (folderName, fileName) => {
                    if (!fileName || fileName === "No files found" || fileName === "Select folder first") {
                        clearImageDisplay();
                        hideCaptionGenerator();
                        return;
                    }
                    
                    const baseName = fileName.replace(/\.[^/.]+$/, "");
                    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'];
                    
                    for (const ext of imageExtensions) {
                        const imageFileName = baseName + ext;
                        
                        try {
                            const response = await api.fetchApi(`/prompt_viewer/read_image?folder=${encodeURIComponent(folderName)}&filename=${encodeURIComponent(imageFileName)}`);
                            
                            if (response.ok) {
                                const blob = await response.blob();
                                const imageDataUrl = URL.createObjectURL(blob);
                                displayImage(imageDataUrl);
                                this.hasAssociatedImage = true;
                                this.currentImagePath = imageFileName;
                                return;
                            }
                        } catch (error) {
                            continue;
                        }
                    }
                    
                    clearImageDisplay();
                    this.hasAssociatedImage = false;
                    this.currentImagePath = null;
                };
                
                const checkCaptionGeneratorNeeded = () => {
                    if (this.hasAssociatedImage && !this.hasCaptionFile) {
                        showCaptionGenerator();
                    } else {
                        hideCaptionGenerator();
                    }
                };
                
                const loadFileContent = async (folderName, fileName) => {
                    if (!fileName || fileName === "No files found" || fileName === "Select folder first") {
                        if (this.contentPreview) {
                            this.contentPreview.value = "No file selected";
                            this.originalContent = this.contentPreview.value;
                            this.isModified = false;
                            this.updateButtonStates();
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            if (this.contentWidget) {
                                this.contentWidget.value = this.contentPreview.value;
                            }
                        }
                        clearImageDisplay();
                        hideCaptionGenerator();
                        return;
                    }
                    
                    this.currentFolder = folderName;
                    this.currentFile = fileName;
                    
                    // Check if the selected file is an image
                    const isImageFile = fileName.toLowerCase().match(/\.(png|jpg|jpeg|gif|bmp|webp)$/);
                    
                    if (this.contentPreview) {
                        this.contentPreview.value = "Loading...";
                    }
                    
                    if (isImageFile) {
                        // This is an image file - try to load it directly
                        this.hasCaptionFile = false;
                        
                        if (this.contentPreview) {
                            this.contentPreview.value = "";
                            this.originalContent = "";
                            this.isModified = false;
                            this.updateButtonStates();
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            if (this.contentWidget) {
                                this.contentWidget.value = "";
                            }
                        }
                        
                        // Load the image directly using its filename
                        try {
                            const response = await api.fetchApi(`/prompt_viewer/read_image?folder=${encodeURIComponent(folderName)}&filename=${encodeURIComponent(fileName)}`);
                            
                            if (response.ok) {
                                const blob = await response.blob();
                                const imageDataUrl = URL.createObjectURL(blob);
                                displayImage(imageDataUrl);
                                this.hasAssociatedImage = true;
                                this.currentImagePath = fileName;
                                checkCaptionGeneratorNeeded();
                                return;
                            }
                        } catch (error) {
                            console.error("Error loading image:", error);
                        }
                        
                        clearImageDisplay();
                        this.hasAssociatedImage = false;
                        checkCaptionGeneratorNeeded();
                        return;
                    }
                    
                    // This is a text file - proceed with normal text loading
                    try {
                        const response = await api.fetchApi(`/prompt_viewer/read?folder=${encodeURIComponent(folderName)}&filename=${encodeURIComponent(fileName)}`);
                        
                        if (response.ok) {
                            const text = await response.text();
                            this.hasCaptionFile = true;
                            
                            if (this.contentPreview) {
                                this.contentPreview.value = text;
                                this.originalContent = text;
                                this.isModified = false;
                                this.updateButtonStates();
                                
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                }
                                
                                if (this.contentWidget) {
                                    this.contentWidget.value = text;
                                }
                                
                                const lines = text.split('\n').length;
                                this.contentPreview.inputEl.rows = Math.min(Math.max(lines, 10), 30);
                                this.setDirtyCanvas(true, true);
                            }
                            
                            await loadAssociatedImage(folderName, fileName);
                            checkCaptionGeneratorNeeded();
                        } else {
                            this.hasCaptionFile = false;
                            
                            if (this.contentPreview) {
                                this.contentPreview.value = "";
                                this.originalContent = "";
                                this.isModified = false;
                                this.updateButtonStates();
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                }
                                if (this.contentWidget) {
                                    this.contentWidget.value = "";
                                }
                            }
                            
                            await loadAssociatedImage(folderName, fileName);
                            checkCaptionGeneratorNeeded();
                        }
                    } catch (error) {
                        this.hasCaptionFile = false;
                        
                        if (this.contentPreview) {
                            this.contentPreview.value = "";
                            this.originalContent = "";
                            this.isModified = false;
                            this.updateButtonStates();
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            if (this.contentWidget) {
                                this.contentWidget.value = "";
                            }
                        }
                        
                        await loadAssociatedImage(folderName, fileName);
                        checkCaptionGeneratorNeeded();
                    }
                };
                
                this.generateCaption = async () => {
    if (!this.hasAssociatedImage || !this.currentImagePath) {
        alert("No image available for caption generation");
        return;
    }
    
    if (this.generateCaptionButton) {
        this.generateCaptionButton.disabled = true;
        this.generateCaptionButton.textContent = "Generating...";
    }
    
    if (this.contentPreview) {
        this.contentPreview.value = "Generating caption using Moondream2...";
    }
    
    try {
        const response = await api.fetchApi("/prompt_viewer/generate_caption", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                folder: this.currentFolder,
                image_filename: this.currentImagePath
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            const caption = result.caption || "";
            
            if (this.contentPreview) {
                this.contentPreview.value = caption;
                this.originalContent = "";
                this.isModified = true;
                this.updateButtonStates();
                
                if (this.editedWidget) {
                    this.editedWidget.value = true;
                    if (this.editedWidget.callback) {
                        this.editedWidget.callback(this.editedWidget.value);
                    }
                }
                
                if (this.contentWidget) {
                    this.contentWidget.value = caption;
                }
            }
            
            // AUTO-SAVE THE CAPTION WITH SAME NAME AS IMAGE FILE
            try {
                const saveResponse = await api.fetchApi("/prompt_viewer/auto_save_caption", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        folder: this.currentFolder,
                        image_filename: this.currentImagePath,
                        caption: caption
                    })
                });
                
                if (saveResponse.ok) {
                    const saveResult = await saveResponse.json();
                    console.log("Caption auto-saved:", saveResult.message);
                    
                    // Update UI to reflect the saved file
                    this.currentFile = saveResult.filename || "";
                    this.hasCaptionFile = true;
                    
                    // Refresh file list to show the new .txt file
                    await updateFileList(this.currentFolder);
                    
                    hideCaptionGenerator();
                    
                    // Auto-select the new text file in the dropdown
                    const fileWidget = this.widgets.find(w => w.name === "file");
                    if (fileWidget && saveResult.filename) {
                        fileWidget.value = saveResult.filename;
                        if (fileWidget.callback) {
                            fileWidget.callback(fileWidget.value);
                        }
                    }
                    
                    alert("Caption generated and saved automatically as: " + saveResult.filename);
                } else {
                    const errorText = await saveResponse.text();
                    console.warn("Auto-save failed:", errorText);
                    alert("Caption generated but auto-save failed. You can save manually.");
                }
            } catch (saveError) {
                console.warn("Auto-save error:", saveError);
                alert("Caption generated but auto-save failed. You can save manually.");
            }
        } else {
            const errorText = await response.text();
            alert("Error generating caption: " + errorText);
            if (this.contentPreview) {
                this.contentPreview.value = "";
            }
        }
    } catch (error) {
        alert("Error generating caption: " + error.message);
        if (this.contentPreview) {
            this.contentPreview.value = "";
        }
    } finally {
        if (this.generateCaptionButton) {
            this.generateCaptionButton.disabled = false;
            this.generateCaptionButton.textContent = "Yes - Generate Caption";
        }
    }
};
                
                this.saveFile = async (saveAs) => {
                    if (!this.contentPreview) {
                        alert("No content to save");
                        return;
                    }
                    
                    let targetFolder = this.currentFolder || "(root)";
                    let targetFile = this.currentFile;
                    
                    if (saveAs || !targetFile) {
                        const folderName = targetFolder === "(root)" ? "root prompts folder" : `folder: ${targetFolder}`;
                        const defaultName = targetFile || "new_file.txt";
                        const newName = prompt(`Enter filename (will be saved in ${folderName}).\nUse folder\\filename.txt to create new folder:`, defaultName);
                        if (!newName) return;
    
                        if (newName.includes('\\') || newName.includes('/')) {
                            const separator = newName.includes('\\') ? '\\' : '/';
                            const parts = newName.split(separator);
        
                            if (parts.length === 2) {
                                const [newFolder, fileName] = parts;
            
                                if (!fileName.match(/\.(txt|log|json|csv|md)$/i)) {
                                    alert("Filename must end with .txt, .log, .json, .csv, or .md");
                                    return;
                                }
            
                                targetFolder = newFolder;
                                targetFile = fileName;
                            } else {
                                alert("Invalid path format. Use: folder\\filename.txt");
                                return;
                            }
                        } else {
                            if (!newName.match(/\.(txt|log|json|csv|md)$/i)) {
                                alert("Filename must end with .txt, .log, .json, .csv, or .md");
                                return;
                            }
                            targetFile = newName;
                        }
                    }
                    
                    if (!targetFile) {
                        alert("No filename provided");
                        return;
                    }
                    
                    const content = this.contentPreview.value;
                    
                    try {
                        const response = await api.fetchApi("/prompt_viewer/save", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                folder: targetFolder,
                                filename: targetFile,
                                content: content
                            })
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            alert(result.message || "File saved successfully");
    
                            this.currentFolder = targetFolder;
                            this.currentFile = targetFile;
                            this.hasCaptionFile = true;
    
                            const folderWidget = this.widgets.find(w => w.name === "folder");
                            const fileWidget = this.widgets.find(w => w.name === "file");
                            if (folderWidget) {
                                if (result.folder_created) {
                                    try {
                                        const foldersResponse = await api.fetchApi('/prompt_viewer/list_folders');
                                        if (foldersResponse.ok) {
                                            const foldersData = await foldersResponse.json();
                                            folderWidget.options.values = foldersData.folders;
                                            folderWidget.value = targetFolder;
                                        }
                                    } catch (error) {
                                        console.error("Error refreshing folder list:", error);
                                    }
                                }
                                
                                try {
                                    const response = await api.fetchApi(`/prompt_viewer/list_files?folder=${encodeURIComponent(targetFolder)}`);
                                    
                                    if (response.ok) {
                                        const data = await response.json();
                                        if (fileWidget) {
                                            fileWidget.options.values = data.files;
                                            fileWidget.value = targetFile;
                                        }
                                    }
                                } catch (error) {
                                    console.error("Error updating file list:", error);
                                }
                            }
    
                            this.originalContent = content;
                            this.isModified = false;
                            this.updateButtonStates();
    
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            
                            hideCaptionGenerator();
                            await loadAssociatedImage(targetFolder, targetFile);
                        } else {
                            const errorText = await response.text();
                            alert("Error saving file: " + errorText);
                        }
                    } catch (error) {
                        alert("Error saving file: " + error.message);
                    }
                };
                
                this.clearContent = () => {
                    if (!this.contentPreview) return;
                    
                    this.contentPreview.value = "";
                    this.originalContent = "";
                    this.isModified = true;
                    this.currentFile = "";
                    
                    this.updateButtonStates();
                    
                    if (this.editedWidget) {
                        this.editedWidget.value = true;
                        if (this.editedWidget.callback) {
                            this.editedWidget.callback(this.editedWidget.value);
                        }
                    }
                    
                    if (this.contentWidget) {
                        this.contentWidget.value = "";
                    }
                    
                    clearImageDisplay();
                    hideCaptionGenerator();
                    this.setDirtyCanvas(true, true);
                };
                
                this.updateButtonStates = () => {
                    if (this.saveButton) {
                        this.saveButton.disabled = !this.isModified;
                    }
                };
                
                const updateFileList = async (folderName) => {
                    try {
                        const response = await api.fetchApi(`/prompt_viewer/list_files?folder=${encodeURIComponent(folderName)}`);
                        
                        if (response.ok) {
                            const data = await response.json();
                            const fileWidget = this.widgets.find(w => w.name === "file");
                            
                            if (fileWidget) {
                                fileWidget.options.values = data.files;
                                fileWidget.value = data.files[0] || "No files found";
                                
                                if (data.files.length > 0 && data.files[0] !== "No files found") {
                                    loadFileContent(folderName, data.files[0]);
                                } else {
                                    if (this.contentPreview) {
                                        this.contentPreview.value = "No files found in this folder";
                                        this.originalContent = this.contentPreview.value;
                                        this.isModified = false;
                                        this.updateButtonStates();
                                        if (this.editedWidget) {
                                            this.editedWidget.value = false;
                                            if (this.editedWidget.callback) {
                                                this.editedWidget.callback(this.editedWidget.value);
                                            }
                                        }
                                        if (this.contentWidget) {
                                            this.contentWidget.value = this.contentPreview.value;
                                        }
                                    }
                                    clearImageDisplay();
                                    hideCaptionGenerator();
                                }
                            }
                        }
                    } catch (error) {
                        console.error("Error updating file list:", error);
                    }
                };
                
                const folderWidget = this.widgets.find(w => w.name === "folder");
                const fileWidget = this.widgets.find(w => w.name === "file");
                
                if (folderWidget) {
                    const originalFolderCallback = folderWidget.callback;
                    
                    folderWidget.callback = (value) => {
                        if (originalFolderCallback) {
                            originalFolderCallback.call(folderWidget, value);
                        }
                        this.currentFolder = value;
                        updateFileList(value);
                    };
                    
                    this.currentFolder = folderWidget.value || "(root)";
                }
                
                if (fileWidget) {
                    const originalFileCallback = fileWidget.callback;
                    
                    fileWidget.callback = (value) => {
                        if (originalFileCallback) {
                            originalFileCallback.call(fileWidget, value);
                        }
                        this.currentFile = value;
                        const currentFolder = folderWidget ? folderWidget.value : "(root)";
                        this.currentFolder = currentFolder;
                        loadFileContent(currentFolder, value);
                    };
                    
                    this.currentFile = fileWidget.value || "";
                }
                
                this.updateButtonStates();
                
                console.log("=== GRPromptViewer onNodeCreated END ===");
                return r;
            };
            
            nodeType.prototype.onExecuted = function(message) {
                originalOnExecuted?.apply(this, arguments);
            };
        }
    }
});