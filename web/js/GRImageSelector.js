import { app } from "/scripts/app.js";

console.log("GRImageSelector extension loading...");

// Register the custom node
app.registerExtension({
    name: "GRImageSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GRImageSelector") {
            // Override the onConnectionsChange method to detect when inputs are connected
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // Only handle input connections
                if (type === 1) { // 1 = input, 2 = output
                    if (connected) {
                        // Check if all current inputs are connected
                        let allConnected = true;
                        for (let i = 0; i < this.inputs.length; i++) {
                            if (!this.inputs[i].link) {
                                allConnected = false;
                                break;
                            }
                        }
                        
                        // If all inputs are connected, add a new one
                        if (allConnected) {
                            const newIndex = this.inputs.length + 1;
                            this.addInput(`image_${newIndex}`, "IMAGE");
                        }
                    } else {
                        // When disconnected, remove empty trailing inputs (keep at least 1)
                        while (this.inputs.length > 1) {
                            const lastInput = this.inputs[this.inputs.length - 1];
                            const secondLastInput = this.inputs[this.inputs.length - 2];
                            
                            // Remove last input if it's empty and second-to-last is also empty
                            if (!lastInput.link && !secondLastInput.link) {
                                this.removeInput(this.inputs.length - 1);
                            } else {
                                break;
                            }
                        }
                    }
                }
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass === "GRImageSelector") {
            // Remove all the default inputs that Python created
            while (node.inputs.length > 0) {
                node.removeInput(0);
            }
            
            // Start with just one input
            node.addInput("image_1", "IMAGE");
        }
    }
});