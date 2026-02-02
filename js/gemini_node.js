import { app } from "../../scripts/app.js";

// List of nodes that should have dynamic image inputs
const DYNAMIC_IMAGE_NODES = [
    "GeminiImageGenerator",
    "GeminiAPI",      // Was GeminiLLMAPI - fixed to match NODE_CLASS_MAPPINGS
    "QwenAPI",        // Was GeminiQwenAPI - fixed to match NODE_CLASS_MAPPINGS
    "ClaudeAPI",      // Was GeminiClaudeAPI - fixed to match NODE_CLASS_MAPPINGS
    "OllamaAPI"       // Was GeminiOllamaAPI - fixed to match NODE_CLASS_MAPPINGS
];

app.registerExtension({
    name: "ComfyUI.GeminiDynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (DYNAMIC_IMAGE_NODES.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Define the inputs we want to manage (image5 and beyond are dynamic)
                const managedInputs = [
                    "image5", "image6", "image7", "image8", "image9",
                    "image10", "image11", "image12", "image13", "image14"
                ];

                // We need to hook into onConnectionsChange
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    onConnectionsChange?.apply(this, arguments);
                    // We defer the update slightly to ensure state is settled
                    setTimeout(() => {
                        this.updateDynamicInputs();
                    }, 20);
                };

                this.updateDynamicInputs = function () {
                    if (!this.inputs) return;

                    // Map of all possible image inputs in order
                    const allImageInputs = [
                        "image1", "image2", "image3", "image4",
                        "image5", "image6", "image7", "image8", "image9",
                        "image10", "image11", "image12", "image13", "image14"
                    ];

                    // Find the index of the last connected image input (among image1-4 minimum)
                    let lastConnectedIndex = -1;
                    for (let i = 0; i < allImageInputs.length; i++) {
                        const name = allImageInputs[i];
                        const inp = this.inputs.find(inpt => inpt.name === name);
                        if (inp && inp.link !== null) {
                            lastConnectedIndex = i;
                        }
                    }

                    // We want to show:
                    // - Always image1-4 (indices 0-3)
                    // - Plus one more if image4 (index 3) is connected
                    // - Continue pattern for subsequent connections
                    let targetIndex;
                    if (lastConnectedIndex < 3) {
                        // No images beyond image3 connected, show only image1-4
                        targetIndex = 3;
                    } else {
                        // Last connected is image4 or beyond, show one more
                        targetIndex = lastConnectedIndex + 1;
                    }

                    // Cap at max index
                    if (targetIndex >= allImageInputs.length) {
                        targetIndex = allImageInputs.length - 1;
                    }

                    // Target inputs to keep visible
                    const targetInputNames = allImageInputs.slice(0, targetIndex + 1);

                    // Remove inputs that shouldn't be there (from end to start to avoid index issues)
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const inp = this.inputs[i];
                        // Only touch our managed image inputs (image5+)
                        if (managedInputs.includes(inp.name)) {
                            if (!targetInputNames.includes(inp.name)) {
                                this.removeInput(i);
                            }
                        }
                    }

                    // Add inputs that should be visible but are missing
                    for (const name of targetInputNames) {
                        // Skip if it's not one of our managed ones (1-4 are static from Python)
                        if (!managedInputs.includes(name)) continue;

                        const exists = this.inputs.find(i => i.name === name);
                        if (!exists) {
                            this.addInput(name, "IMAGE");
                        }
                    }
                };

                // Initial update - remove extra inputs immediately after node creation
                setTimeout(() => {
                    this.updateDynamicInputs();
                }, 50);
            };
        }
    },
});
