import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "GRPromptViewerExtension",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GRPromptViewer") {
            console.log("=== Registering GRPromptViewer extension ===");

            // Store the original node prototype
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            const originalOnExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onNodeCreated = function() {
                console.log("=== GRPromptViewer onNodeCreated START ===");
                
                const r = originalNodeCreated ? originalNodeCreated.apply(this, arguments) : undefined;
                
                console.log("Widgets after original onNodeCreated:", this.widgets);
                
                // Find the content input widget
                this.contentWidget = this.widgets.find(w => w.name === "content");
                this.editedWidget = this.widgets.find(w => w.name === "edited");
                
                // Debug widgets
                console.log("Content widget:", this.contentWidget);
                console.log("Edited widget:", this.editedWidget);
                
                // Hide the content widget if it exists
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
                    // Ensure edited widget is properly initialized as boolean false
                    this.editedWidget.value = false;
                    console.log("Initialized edited widget value:", this.editedWidget.value, "type:", typeof this.editedWidget.value);
                }
                
                // Create preview widget
                try {
                    console.log("Creating preview widget...");
                    
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
                        
                        // Track changes to enable/disable save button
                        previewWidget.inputEl.addEventListener('input', () => {
                            const isNowModified = (previewWidget.value !== this.originalContent);
                            
                            if (this.isModified !== isNowModified) {
                                this.isModified = isNowModified;
                                console.log("Content modification state changed to:", this.isModified);
                            }
                            
                            // Update save button state
                            if (this.saveButton) {
                                this.saveButton.disabled = !this.isModified;
                            }
                            
                            // Update edited flag
                            if (this.editedWidget) {
                                this.editedWidget.value = Boolean(this.isModified);
                                console.log("Edited flag set to:", this.editedWidget.value);
                                
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            
                            // CRITICAL: Update the content widget with preview value
                            if (this.contentWidget) {
                                this.contentWidget.value = previewWidget.value;
                            }
                            
                            console.log("Content modified:", this.isModified, "Length:", previewWidget.value.length);
                            
                            // Mark node as dirty to force re-execution
                            this.setDirtyCanvas(true, true);
                        });
                        
                        console.log("Preview widget created successfully");
                    } else {
                        console.error("Failed to create preview widget");
                    }
                } catch (e) {
                    console.error("Error creating preview widget:", e);
                }
                
                // Create HTML container for buttons
                const buttonContainer = document.createElement("div");
                buttonContainer.style.display = "flex";
                buttonContainer.style.gap = "10px";
                buttonContainer.style.padding = "10px";
                buttonContainer.style.width = "100%";
                buttonContainer.style.boxSizing = "border-box";
                
                // Save button
                const saveBtn = document.createElement("button");
                saveBtn.textContent = "Save";
                saveBtn.style.flex = "1";
                saveBtn.style.height = "35px";
                saveBtn.style.padding = "0";
                saveBtn.style.cursor = "pointer";
                saveBtn.style.fontSize = "14px";
                saveBtn.disabled = true;
                saveBtn.onclick = () => {
                    console.log("Save button clicked");
                    this.saveFile(false);
                };
                
                // Save As button
                const saveAsBtn = document.createElement("button");
                saveAsBtn.textContent = "Save As";
                saveAsBtn.style.flex = "1";
                saveAsBtn.style.height = "35px";
                saveAsBtn.style.padding = "0";
                saveAsBtn.style.cursor = "pointer";
                saveAsBtn.style.fontSize = "14px";
                saveAsBtn.onclick = () => {
                    console.log("Save As button clicked");
                    this.saveFile(true);
                };
                
                // Clear button
                const clearBtn = document.createElement("button");
                clearBtn.textContent = "Clear";
                clearBtn.style.flex = "1";
                clearBtn.style.height = "35px";
                clearBtn.style.padding = "0";
                clearBtn.style.cursor = "pointer";
                clearBtn.style.fontSize = "14px";
                clearBtn.onclick = () => {
                    console.log("Clear button clicked");
                    this.clearContent();
                };
                
                buttonContainer.appendChild(saveBtn);
                buttonContainer.appendChild(saveAsBtn);
                buttonContainer.appendChild(clearBtn);
                
                // Add HTML widget with minimal size
                const htmlWidget = this.addDOMWidget("buttons", "div", buttonContainer, {
                    getValue: () => null,
                    setValue: () => {},
                });
                htmlWidget.computeSize = () => [0, 55]; // Fixed small height
                
                this.saveButton = saveBtn;
                this.saveAsButton = saveAsBtn;
                this.clearButton = clearBtn;
                
                // Helper function to check if an input is connected
                this.isInputConnected = function(inputName) {
                    if (!this.inputs) return false;
                    const input = this.inputs.find(i => i.name === inputName);
                    return input && input.link != null;
                };
                
                // Override getInputData to return correct content based on connection state
                const originalGetInputData = this.getInputData;
                this.getInputData = function(slot) {
                    // Check if this is the content input slot
                    const input = this.inputs?.[slot];
                    if (input && input.name === "content") {
                        // If content input is connected, use the original behavior (get from connected node)
                        if (input.link != null) {
                            console.log("Content input connected - using linked node data");
                            return originalGetInputData ? originalGetInputData.call(this, slot) : undefined;
                        }
                        // If content input is NOT connected, use the preview content
                        else if (this.contentPreview) {
                            console.log("Content input NOT connected - using preview content, length:", this.contentPreview.value.length);
                            return this.contentPreview.value;
                        }
                    }
                    return originalGetInputData ? originalGetInputData.call(this, slot) : undefined;
                };
                
                // Function to update file list based on selected folder
                const updateFileList = async (folderName) => {
                    console.log("Updating file list for folder:", folderName);
                    
                    try {
                        const response = await api.fetchApi(`/prompt_viewer/list_files?folder=${encodeURIComponent(folderName)}`);
                        
                        if (response.ok) {
                            const data = await response.json();
                            console.log("Files received:", data.files);
                            
                            if (fileWidget) {
                                fileWidget.options.values = data.files;
                                fileWidget.value = data.files[0] || "No files found";
                                
                                // Trigger file load if there are files
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
                                        // Update content widget
                                        if (this.contentWidget) {
                                            this.contentWidget.value = this.contentPreview.value;
                                        }
                                    }
                                    // Clear image display
                                    clearImageDisplay();
                                }
                            }
                        }
                    } catch (error) {
                        console.error("Error updating file list:", error);
                    }
                };
                
                // Function to clear image display
                const clearImageDisplay = () => {
                    if (this.imageDisplayContainer) {
                        this.imageDisplayContainer.style.display = "none";
                        this.imageDisplayContainer.innerHTML = "";
                    }
                    if (this.imageWidget) {
                        this.imageWidget.computeSize = () => [0, 0];
                    }
                };
                
                // Function to display image
                const displayImage = (imageDataUrl) => {
                    if (!this.imageDisplayContainer) {
                        // Create image display container
                        this.imageDisplayContainer = document.createElement("div");
                        this.imageDisplayContainer.style.padding = "10px";
                        this.imageDisplayContainer.style.textAlign = "center";
                        this.imageDisplayContainer.style.borderTop = "1px solid #444";
                        this.imageDisplayContainer.style.marginTop = "10px";
                        
                        // Create image widget
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
                    
                    // Set image widget size
                    this.imageWidget.computeSize = () => [0, 420];
                    
                    console.log("Image displayed");
                };
                
                // Function to check and load associated image
                const loadAssociatedImage = async (folderName, fileName) => {
                    if (!fileName || fileName === "No files found" || fileName === "Select folder first") {
                        clearImageDisplay();
                        return;
                    }
                    
                    // Get base filename without extension
                    const baseName = fileName.replace(/\.[^/.]+$/, "");
                    
                    // Supported image extensions
                    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'];
                    
                    // Try each extension
                    for (const ext of imageExtensions) {
                        const imageFileName = baseName + ext;
                        
                        try {
                            const response = await api.fetchApi(`/prompt_viewer/read_image?folder=${encodeURIComponent(folderName)}&filename=${encodeURIComponent(imageFileName)}`);
                            
                            if (response.ok) {
                                const blob = await response.blob();
                                const imageDataUrl = URL.createObjectURL(blob);
                                displayImage(imageDataUrl);
                                console.log("Found associated image:", imageFileName);
                                return;
                            }
                        } catch (error) {
                            // Continue to next extension
                            continue;
                        }
                    }
                    
                    // No image found
                    clearImageDisplay();
                    console.log("No associated image found for:", fileName);
                };
                
                // Function to load file content
                const loadFileContent = async (folderName, fileName) => {
                    console.log("Loading file:", fileName, "from folder:", folderName);
                    
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
                            // Update content widget
                            if (this.contentWidget) {
                                this.contentWidget.value = this.contentPreview.value;
                            }
                        }
                        clearImageDisplay();
                        return;
                    }
                    
                    // Store current file info
                    this.currentFolder = folderName;
                    this.currentFile = fileName;
                    
                    if (this.contentPreview) {
                        this.contentPreview.value = "Loading...";
                    }
                    
                    try {
                        const response = await api.fetchApi(`/prompt_viewer/read?folder=${encodeURIComponent(folderName)}&filename=${encodeURIComponent(fileName)}`);
                        
                        if (response.ok) {
                            const text = await response.text();
                            console.log("Loaded file, length:", text.length);
                            
                            if (this.contentPreview) {
                                this.contentPreview.value = text;
                                this.originalContent = text;
                                this.isModified = false;
                                
                                this.updateButtonStates();
                                
                                // Reset edited flag when loading a NEW file
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    console.log("Reset edited flag to false after loading file");
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                }
                                
                                // CRITICAL: Update content widget with loaded text
                                if (this.contentWidget) {
                                    this.contentWidget.value = text;
                                    console.log("Updated content widget with file content, length:", text.length);
                                }
                                
                                const lines = text.split('\n').length;
                                this.contentPreview.inputEl.rows = Math.min(Math.max(lines, 10), 30);
                                
                                this.setDirtyCanvas(true, true);
                            }
                            
                            // Try to load associated image
                            loadAssociatedImage(folderName, fileName);
                        } else {
                            const errorText = await response.text();
                            if (this.contentPreview) {
                                this.contentPreview.value = `Error: ${errorText}`;
                                this.originalContent = this.contentPreview.value;
                                this.isModified = false;
                                this.updateButtonStates();
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                }
                                // Update content widget
                                if (this.contentWidget) {
                                    this.contentWidget.value = this.contentPreview.value;
                                }
                            }
                            clearImageDisplay();
                        }
                    } catch (error) {
                        console.error("Error fetching file:", error);
                        if (this.contentPreview) {
                            this.contentPreview.value = "Error loading file: " + error.message;
                            this.originalContent = this.contentPreview.value;
                            this.isModified = false;
                            this.updateButtonStates();
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            // Update content widget
                            if (this.contentWidget) {
                                this.contentWidget.value = this.contentPreview.value;
                            }
                        }
                        clearImageDisplay();
                    }
                };
                
                // Function to save file
                this.saveFile = async (saveAs) => {
                    if (!this.contentPreview) {
                        alert("No content to save");
                        return;
                    }
                    
                    let targetFolder = this.currentFolder || "(root)";
                    let targetFile = this.currentFile;
                    
                    // If no current file or saveAs, prompt for filename
                    if (saveAs || !targetFile) {
                        const folderName = targetFolder === "(root)" ? "root prompts folder" : `folder: ${targetFolder}`;
                        const defaultName = targetFile || "new_file.txt";
                        const newName = prompt(`Enter filename (will be saved in ${folderName}).\nUse folder\\filename.txt to create new folder:`, defaultName);
                        if (!newName) return; // User cancelled
    
                        // Check if user included a folder path (e.g., "modified\face001.txt")
                        if (newName.includes('\\') || newName.includes('/')) {
                            const separator = newName.includes('\\') ? '\\' : '/';
                            const parts = newName.split(separator);
        
                            if (parts.length === 2) {
                                const [newFolder, fileName] = parts;
            
                                // Ensure filename has a valid extension
                                if (!fileName.match(/\.(txt|log|json|csv|md)$/i)) {
                                    alert("Filename must end with .txt, .log, .json, .csv, or .md");
                                    return;
                                }
            
                                // Update target folder and file
                                targetFolder = newFolder;
                                targetFile = fileName;
                            } else {
                                alert("Invalid path format. Use: folder\\filename.txt");
                                return;
                            }
                        } else {
                            // Ensure it has a valid extension
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
                    
                    console.log("Saving file:", targetFile, "to folder:", targetFolder);
                    
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
    
                            // Update current file info
                            this.currentFolder = targetFolder;
                            this.currentFile = targetFile;
    
                            // Refresh file list to show new file
                            const folderWidget = this.widgets.find(w => w.name === "folder");
                            const fileWidget = this.widgets.find(w => w.name === "file");
                            if (folderWidget) {
                                // If we created a new folder, update the folder widget options first
                                if (result.folder_created) {
                                    // Fetch updated folder list
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
                                
                                // Update file list WITHOUT auto-loading
                                try {
                                    const response = await api.fetchApi(`/prompt_viewer/list_files?folder=${encodeURIComponent(targetFolder)}`);
                                    
                                    if (response.ok) {
                                        const data = await response.json();
                                        console.log("Files received:", data.files);
                                        
                                        if (fileWidget) {
                                            fileWidget.options.values = data.files;
                                            // Set the file widget to the saved file, not the first file
                                            fileWidget.value = targetFile;
                                        }
                                    }
                                } catch (error) {
                                    console.error("Error updating file list:", error);
                                }
                            }
    
                            // Mark as unmodified
                            this.originalContent = content;
                            this.isModified = false;
                            this.updateButtonStates();
    
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                            }
                            
                            // Try to load associated image for the saved file
                            loadAssociatedImage(targetFolder, targetFile);
                        } else {
                            const errorText = await response.text();
                            alert("Error saving file: " + errorText);
                        }
                    } catch (error) {
                        console.error("Error saving file:", error);
                        alert("Error saving file: " + error.message);
                    }
                };
                
                // Function to clear content
                this.clearContent = () => {
                    if (!this.contentPreview) {
                        return;
                    }
                    
                    // Clear the text area
                    this.contentPreview.value = "";
                    this.originalContent = "";
                    this.isModified = true;
                    this.currentFile = "";
                    
                    console.log("Content cleared, folder preserved:", this.currentFolder);
                    
                    this.updateButtonStates();
                    
                    if (this.editedWidget) {
                        this.editedWidget.value = true;
                        if (this.editedWidget.callback) {
                            this.editedWidget.callback(this.editedWidget.value);
                        }
                    }
                    
                    // Update content widget
                    if (this.contentWidget) {
                        this.contentWidget.value = "";
                    }
                    
                    // Clear image display
                    clearImageDisplay();
                    
                    this.setDirtyCanvas(true, true);
                };
                
                // Helper function to update button states
                this.updateButtonStates = () => {
                    if (this.saveButton) {
                        this.saveButton.disabled = !this.isModified;
                    }
                };
                
                // Find the folder and file widgets
                const folderWidget = this.widgets.find(w => w.name === "folder");
                const fileWidget = this.widgets.find(w => w.name === "file");
                
                console.log("Folder widget found:", folderWidget);
                console.log("File widget found:", fileWidget);
                
                // Hook folder widget callback
                if (folderWidget) {
                    const originalFolderCallback = folderWidget.callback;
                    const node = this;
                    
                    folderWidget.callback = function(value) {
                        if (originalFolderCallback) {
                            originalFolderCallback.call(folderWidget, value);
                        }
                        
                        console.log("Folder selected:", value);
                        node.currentFolder = value;
                        console.log("node.currentFolder set to:", node.currentFolder);
                        updateFileList(value);
                    };
                    
                    // Set initial folder value
                    this.currentFolder = folderWidget.value || "(root)";
                    console.log("Initial folder set to:", this.currentFolder);
                    
                    console.log("Folder callback hooked");
                }
                
                // Hook file widget callback
                if (fileWidget) {
                    const originalFileCallback = fileWidget.callback;
                    const node = this;
                    
                    fileWidget.callback = function(value) {
                        if (originalFileCallback) {
                            originalFileCallback.call(fileWidget, value);
                        }
                        
                        console.log("File selected:", value);
                        node.currentFile = value;
                        
                        const currentFolder = folderWidget ? folderWidget.value : "(root)";
                        node.currentFolder = currentFolder;
                        console.log("node.currentFolder updated to:", node.currentFolder);
                        console.log("node.currentFile updated to:", node.currentFile);
                        loadFileContent(currentFolder, value);
                    };
                    
                    // Set initial file value
                    this.currentFile = fileWidget.value || "";
                    console.log("Initial file set to:", this.currentFile);
                    
                    console.log("File callback hooked");
                }
                
                // Initialize button states
                this.updateButtonStates();
                
                console.log("=== GRPromptViewer onNodeCreated END ===");
                return r;
            };
            
            // Handle execution updates
            nodeType.prototype.onExecuted = function(message) {
                originalOnExecuted?.apply(this, arguments);
                console.log("Node executed, message:", message);
                
                // DON'T reset the edited flag or sync content after execution
                // The preview content is the source of truth
                console.log("Execution complete, preserving current state");
            };
        }
    }
});