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
                
                // Find the content input widget (it's now an optional input, not hidden)
                this.contentWidget = this.widgets.find(w => w.name === "content");
                this.editedWidget = this.widgets.find(w => w.name === "edited");
                
                // Debug widgets
                console.log("Content widget:", this.contentWidget);
                console.log("Edited widget:", this.editedWidget);
                
                // Hide the content widget if it exists (since we have content_preview)
                if (this.contentWidget) {
                    this.contentWidget.computeSize = () => [0, -4];
                    if (this.contentWidget.inputEl) {
                        this.contentWidget.inputEl.style.display = "none";
                    }
                }
                
                // Hide the edited widget
                if (this.editedWidget) {
                    this.editedWidget.computeSize = () => [0, -4];
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
                        this.contentFromInput = false; // Track if content came from input connection
                        
                        // Track changes to enable/disable save button and sync content
                        previewWidget.inputEl.addEventListener('input', () => {
                            const isNowModified = (previewWidget.value !== this.originalContent);
                            
                            // Only update if changed to avoid infinite loops
                            if (this.isModified !== isNowModified) {
                                this.isModified = isNowModified;
                                console.log("Content modification state changed to:", this.isModified);
                            }
                            
                            // Update save button state
                            if (this.saveButton) {
                                this.saveButton.disabled = !this.isModified;
                            }
                            
                            // Sync content to content widget if it exists (not connected)
                            if (this.contentWidget && !this.isInputConnected("content")) {
                                this.contentWidget.value = previewWidget.value;
                                console.log("Synced content to content widget, length:", previewWidget.value.length);
                            }
                            
                            // Update edited flag - THIS IS CRITICAL FOR AUTO-SAVE
                            if (this.editedWidget) {
                                // Force boolean value
                                this.editedWidget.value = Boolean(this.isModified);
                                console.log("Edited flag set to:", this.editedWidget.value, "(type:", typeof this.editedWidget.value + ")");
                                
                                // Force the widget to update the node's input value
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
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
                
                // Override onExecute to ensure proper synchronization
                const originalOnExecute = this.onExecute;
                this.onExecute = function() {
                    console.log("=== onExecute called ===");
                    console.log("Current state - isModified:", this.isModified, "content length:", this.contentPreview?.value.length);
                    console.log("Content input connected:", this.isInputConnected("content"));
                    
                    // Check if content is coming from an input connection
                    const contentConnected = this.isInputConnected("content");
                    
                    // If content input is connected, don't sync from preview
                    // The connected input will provide the content
                    if (!contentConnected) {
                        // CRITICAL: Ensure all widget values are properly synchronized
                        // Sync preview content to content widget
                        if (this.contentPreview && this.contentWidget) {
                            this.contentWidget.value = this.contentPreview.value;
                            console.log("Synced content to widget, length:", this.contentPreview.value.length);
                            
                            // Force content widget callback
                            if (this.contentWidget.callback) {
                                this.contentWidget.callback(this.contentWidget.value);
                            }
                        }
                        
                        // Sync edited flag - THIS TRIGGERS AUTO-SAVE for manual edits
                        if (this.editedWidget) {
                            // Force boolean value
                            this.editedWidget.value = Boolean(this.isModified);
                            console.log("Synced edited flag for execution:", this.editedWidget.value, "(type:", typeof this.editedWidget.value + ")");
                            
                            // Force edited widget callback
                            if (this.editedWidget.callback) {
                                this.editedWidget.callback(this.editedWidget.value);
                            }
                        }
                    } else {
                        console.log("Content input is connected - will use input value, not preview");
                        // Reset edited flag when content comes from input
                        if (this.editedWidget) {
                            this.editedWidget.value = false;
                            if (this.editedWidget.callback) {
                                this.editedWidget.callback(this.editedWidget.value);
                            }
                            console.log("Reset edited flag - content from input connection");
                        }
                    }
                    
                    // Force the node to update its inputs
                    if (this.onInputChanged) {
                        this.onInputChanged();
                    }
                    
                    // Mark the node as dirty to ensure execution
                    this.setDirtyCanvas(true, true);
                    
                    if (originalOnExecute) {
                        return originalOnExecute.apply(this, arguments);
                    }
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
                                        // Reset edited flag when no files
                                        if (this.editedWidget) {
                                            this.editedWidget.value = false;
                                            if (this.editedWidget.callback) {
                                                this.editedWidget.callback(this.editedWidget.value);
                                            }
                                            console.log("Reset edited flag - no files found");
                                        }
                                    }
                                }
                            }
                        }
                    } catch (error) {
                        console.error("Error updating file list:", error);
                    }
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
                            // Reset edited flag when no file selected
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                                console.log("Reset edited flag - no file selected");
                            }
                        }
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
                                
                                // Reset edited flag and sync content when loading new file
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                    console.log("Reset edited flag after loading file");
                                }
                                if (this.contentWidget && !this.isInputConnected("content")) {
                                    this.contentWidget.value = text;
                                    if (this.contentWidget.callback) {
                                        this.contentWidget.callback(this.contentWidget.value);
                                    }
                                    console.log("Synced content widget after loading file");
                                }
                                
                                const lines = text.split('\n').length;
                                this.contentPreview.inputEl.rows = Math.min(Math.max(lines, 10), 30);
                                
                                this.setDirtyCanvas(true, true);
                            }
                        } else {
                            const errorText = await response.text();
                            if (this.contentPreview) {
                                this.contentPreview.value = `Error: ${errorText}`;
                                this.originalContent = this.contentPreview.value;
                                this.isModified = false;
                                this.updateButtonStates();
                                // Reset edited flag on error
                                if (this.editedWidget) {
                                    this.editedWidget.value = false;
                                    if (this.editedWidget.callback) {
                                        this.editedWidget.callback(this.editedWidget.value);
                                    }
                                    console.log("Reset edited flag - error loading file");
                                }
                            }
                        }
                    } catch (error) {
                        console.error("Error fetching file:", error);
                        if (this.contentPreview) {
                            this.contentPreview.value = "Error loading file: " + error.message;
                            this.originalContent = this.contentPreview.value;
                            this.isModified = false;
                            this.updateButtonStates();
                            // Reset edited flag on error
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                                console.log("Reset edited flag - fetch error");
                            }
                        }
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
                        const newName = prompt(`Enter filename (will be saved in ${folderName}):`, defaultName);
                        if (!newName) return; // User cancelled
                        
                        // Ensure it has a valid extension
                        if (!newName.match(/\.(txt|log|json|csv|md)$/i)) {
                            alert("Filename must end with .txt, .log, .json, .csv, or .md");
                            return;
                        }
                        
                        targetFile = newName;
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
                            this.currentFile = targetFile;
                            
                            // Refresh file list to show new file
                            const folderWidget = this.widgets.find(w => w.name === "folder");
                            if (folderWidget) {
                                updateFileList(folderWidget.value);
                            }
                            
                            // Mark as unmodified
                            this.originalContent = content;
                            this.isModified = false;
                            this.updateButtonStates();
                            
                            // Reset edited flag after save
                            if (this.editedWidget) {
                                this.editedWidget.value = false;
                                if (this.editedWidget.callback) {
                                    this.editedWidget.callback(this.editedWidget.value);
                                }
                                console.log("Reset edited flag after manual save");
                            }
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
                    this.isModified = true; // Mark as modified so save button activates
                    
                    // Keep the current folder but clear the file name
                    // So Save As will prompt for a new name in the same folder
                    this.currentFile = "";
                    
                    console.log("Content cleared, folder preserved:", this.currentFolder);
                    
                    this.updateButtonStates();
                    
                    // Mark as edited and sync - THIS WILL TRIGGER AUTO-SAVE
                    if (this.editedWidget) {
                        this.editedWidget.value = true;
                        if (this.editedWidget.callback) {
                            this.editedWidget.callback(this.editedWidget.value);
                        }
                        console.log("Set edited flag after clear for auto-save");
                    }
                    if (this.contentWidget && !this.isInputConnected("content")) {
                        this.contentWidget.value = "";
                        if (this.contentWidget.callback) {
                            this.contentWidget.callback(this.contentWidget.value);
                        }
                        console.log("Cleared content widget");
                    }
                    
                    // Force node update
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
                
                // After execution, update the original content to match current
                if (this.contentPreview) {
                    // Check if content came from an input connection
                    const contentConnected = this.isInputConnected("content");
                    
                    if (contentConnected) {
                        console.log("Execution completed with content from input - auto-save should have triggered");
                    }
                    
                    this.originalContent = this.contentPreview.value;
                    this.isModified = false;
                    this.updateButtonStates();
                    
                    // Reset edited flag after execution (content has been auto-saved if needed)
                    if (this.editedWidget) {
                        this.editedWidget.value = false;
                        if (this.editedWidget.callback) {
                            this.editedWidget.callback(this.editedWidget.value);
                        }
                        console.log("Reset edited flag after execution");
                    }
                    
                    console.log("Content synchronized after execution, length:", this.originalContent.length);
                }
            };
        }
    }
});
