import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.GeminiImageGenerator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiImageGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Define the inputs we want to manage
                const managedInputs = [
                    "image5", "image6", "image7", "image8", "image9",
                    "image10", "image11", "image12", "image13", "image14"
                ];

                // Function to update input visibility
                const updateVisibility = () => {
                    if (!this.inputs) return;

                    // Always show image1-4 (indices 0-3 usually, but we check names)
                    // We only manage image5+

                    // Find the index of image4 to start checking from
                    let lastVisibleIndex = -1;

                    // Helper to find input index by name
                    const findInputIndex = (name) => this.inputs.findIndex(i => i.name === name);

                    // Check image4 connection status
                    const image4Index = findInputIndex("image4");
                    if (image4Index === -1) return; // Should not happen

                    // Logic:
                    // If image4 is connected, show image5.
                    // If image5 is connected, show image6.
                    // ...

                    // We iterate through our managed list
                    let previousInputName = "image4";

                    for (const inputName of managedInputs) {
                        const inputIndex = findInputIndex(inputName);
                        const prevInputIndex = findInputIndex(previousInputName);

                        if (inputIndex === -1 || prevInputIndex === -1) continue;

                        const prevInput = this.inputs[prevInputIndex];
                        const currentInput = this.inputs[inputIndex];

                        // If previous input has a link, show this one
                        // Also show if this one already has a link (to avoid hiding connected inputs)
                        const shouldBeVisible = (prevInput.link !== null) || (currentInput.link !== null);

                        // ComfyUI doesn't have a simple "hide" flag for inputs in the standard API
                        // But we can rename them or move them off-screen? 
                        // Actually, the standard way is to remove the slot if unused, and add it back if needed.
                        // BUT removing slots disconnects them if they were connected (which is fine if we check link !== null)

                        // However, removing/adding slots changes indices and can be messy.
                        // A cleaner way often used is to change the type to hidden or similar, but ComfyUI is strict.

                        // Let's try the "remove if hidden, add if visible" approach.
                        // But wait, if we remove it, we can't check if it's connected later?
                        // No, if it's removed, it's not connected.

                        // Better approach for "Hiding":
                        // We can't easily hide inputs without removing them.
                        // So we will implement: "Add Next Input" logic?
                        // Or just "Hide unused inputs > 4"?

                        // Let's try a simpler visual approach if possible? No.

                        // IMPLEMENTATION:
                        // We will use the `onConnectionsChange` event to trigger updates.
                        // If image4 is connected, we ensure image5 exists.
                        // If image4 is disconnected, and image5 is disconnected, we remove image5.
                    }
                };

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

                    // We want to ensure that for every connected image input (starting from 4), 
                    // the NEXT one is available.
                    // And any inputs beyond the "next empty one" are removed.

                    // Map of all possible image inputs in order
                    const allImageInputs = [
                        "image1", "image2", "image3", "image4",
                        "image5", "image6", "image7", "image8", "image9",
                        "image10", "image11", "image12", "image13", "image14"
                    ];

                    // 1. Determine which inputs SHOULD exist
                    // Always keep 1-4
                    let activeCount = 4;

                    // Check connections to see how far we need to go
                    // We scan from 4 upwards. If 4 is connected, we need 5. If 5 connected, need 6.
                    for (let i = 3; i < allImageInputs.length - 1; i++) {
                        const currentName = allImageInputs[i];
                        const currentInput = this.inputs.find(inp => inp.name === currentName);

                        if (currentInput && currentInput.link !== null) {
                            // Current is connected, so we need the next one
                            activeCount = i + 2; // +1 for 0-index, +1 for next item
                        } else {
                            // Current is NOT connected.
                            // But wait, what if image 6 is connected but 5 is not? 
                            // We should probably keep 5 visible to allow bridging?
                            // Or just keep everything up to the last connected one + 1.
                        }
                    }

                    // Find the index of the last connected image input
                    let lastConnectedIndex = -1;
                    for (let i = 0; i < allImageInputs.length; i++) {
                        const name = allImageInputs[i];
                        const inp = this.inputs.find(inpt => inpt.name === name);
                        if (inp && inp.link !== null) {
                            lastConnectedIndex = i;
                        }
                    }

                    // We want to show up to lastConnectedIndex + 1 (the next empty slot)
                    // But at minimum show up to image4 (index 3)
                    let targetIndex = lastConnectedIndex + 1;
                    if (targetIndex < 3) targetIndex = 3;
                    if (targetIndex >= allImageInputs.length) targetIndex = allImageInputs.length - 1;

                    // So we want inputs 0 to targetIndex to exist.
                    // And inputs targetIndex+1 onwards to be removed.

                    // 2. Sync the inputs
                    const targetInputNames = allImageInputs.slice(0, targetIndex + 1);

                    // Remove inputs that shouldn't be there
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const inp = this.inputs[i];
                        // Only touch our managed image inputs (image5+)
                        if (managedInputs.includes(inp.name)) {
                            if (!targetInputNames.includes(inp.name)) {
                                this.removeInput(i);
                            }
                        }
                    }

                    // Add inputs that are missing
                    for (const name of targetInputNames) {
                        // Skip if it's not one of our managed ones (1-4 are static)
                        if (!managedInputs.includes(name)) continue;

                        const exists = this.inputs.find(i => i.name === name);
                        if (!exists) {
                            // Add it!
                            // We need to know the type. It's "IMAGE".
                            this.addInput(name, "IMAGE");
                        }
                    }

                    // Optional: Sort inputs to ensure order? 
                    // ComfyUI usually appends to the end. 
                    // If we remove image5 and add it back, it goes to the bottom.
                    // This might be annoying if other inputs exist.
                    // But standard GeminiImageGenerator only has these inputs + seed etc.
                    // Inputs are usually drawn in order.
                };

                // Initial update
                setTimeout(() => {
                    this.updateDynamicInputs();
                }, 100);
            };
        }
    },
});
