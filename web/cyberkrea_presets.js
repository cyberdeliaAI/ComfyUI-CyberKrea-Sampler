import { app } from "../../scripts/app.js";

const PRESETS = {
    fast: {
        steps: 8,
        sampler: "euler",
        restart_frac: 0.25,
        sigma_r: 0.65,
        plunge: true,
        detail: 0.50,
        eta0: 1.0,
        sigma_gate: 0.10,
        contraction: 0.70,
    },
    balanced: {
        steps: 12,
        sampler: "euler",
        restart_frac: 0.25,
        sigma_r: 0.65,
        plunge: true,
        detail: 0.60,
        eta0: 1.0,
        sigma_gate: 0.10,
        contraction: 0.70,
    },
    quality: {
        steps: 16,
        sampler: "euler_2m",
        restart_frac: 0.25,
        sigma_r: 0.65,
        plunge: true,
        detail: 0.70,
        eta0: 1.0,
        sigma_gate: 0.10,
        contraction: 0.70,
    },
};

function applyPreset(node, presetName) {
    const values = PRESETS[presetName];
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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const preset = this.widgets?.find((item) => item.name === "preset");
            if (!preset) return result;

            const originalCallback = preset.callback;
            preset.callback = (value, ...args) => {
                const callbackResult = originalCallback?.call(preset, value, ...args);
                applyPreset(this, value);
                return callbackResult;
            };
            return result;
        };
    },
});
