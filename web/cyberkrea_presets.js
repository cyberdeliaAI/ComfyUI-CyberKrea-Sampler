import { app } from "../../scripts/app.js";

function applyPreset(node, presetName, presets) {
    const values = presets[presetName];
    if (!values) return;

    for (const [name, value] of Object.entries(values)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget) widget.value = value;
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "CyberKreaSampler.PresetValues",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CyberKreaSampler") return;
        const presets = nodeData.input?.required?.preset?.[1]?.cyberkrea_presets;
        if (!presets) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const preset = this.widgets?.find((item) => item.name === "preset");
            if (!preset) return result;

            const originalCallback = preset.callback;
            preset.callback = (value, ...args) => {
                const callbackResult = originalCallback?.call(preset, value, ...args);
                applyPreset(this, value, presets);
                return callbackResult;
            };
            return result;
        };
    },
});
