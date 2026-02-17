import { app } from "/scripts/app.js";

console.log("GRAnySelector extension loading...");

app.registerExtension({
    name: "GRAnySelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GRAnySelector") return;

        const original = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onConnectionsChange = function (
            type,
            index,
            connected,
            link_info
        ) {
            if (original) {
                original.apply(this, arguments);
            }

            // 1 = input
            if (type !== 1) return;

            if (connected) {
                const allConnected = this.inputs.every(i => i.link);
                if (allConnected) {
                    const nextIndex = this.inputs.length + 1;
                    this.addInput(`Any_${nextIndex}`, "*");
                }
            } else {
                // trim trailing empty inputs, keep at least one
                while (this.inputs.length > 1) {
                    const last = this.inputs[this.inputs.length - 1];
                    const prev = this.inputs[this.inputs.length - 2];

                    if (!last.link && !prev.link) {
                        this.removeInput(this.inputs.length - 1);
                    } else {
                        break;
                    }
                }
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "GRAnySelector") return;

        // remove Python-created inputs
        while (node.inputs.length > 0) {
            node.removeInput(0);
        }

        // start with one ANY input
        node.addInput("Any_1", "*");
    }
});
