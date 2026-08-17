import { app } from "../../scripts/app.js";

const RESOLUTIONS = {
    "S (~1.0 MP)": [
        "1024x1024 (1:1)",
        "1152x864 (4:3)",
        "864x1152 (3:4)",
        "1344x896 (3:2)",
        "896x1344 (2:3)",
        "1344x768 (16:9)",
        "768x1344 (9:16)",
    ],
    "M (~1.4 MP)": [
        "1184x1184 (1:1)",
        "1344x1008 (4:3)",
        "1008x1344 (3:4)",
        "1568x1040 (3:2)",
        "1040x1568 (2:3)",
        "1568x880 (16:9)",
        "880x1568 (9:16)",
    ],
    "L (~1.7 MP)": [
        "1312x1312 (1:1)",
        "1504x1120 (4:3)",
        "1120x1504 (3:4)",
        "1600x1088 (3:2)",
        "1088x1600 (2:3)",
        "1728x960 (16:9)",
        "960x1728 (9:16)",
    ],
    "XL (~2.1 MP)": [
        "1440x1440 (1:1)",
        "1664x1248 (4:3)",
        "1248x1664 (3:4)",
        "1776x1184 (3:2)",
        "1184x1776 (2:3)",
        "1920x1088 (16:9)",
        "1088x1920 (9:16)",
    ],
};

app.registerExtension({
    name: "CyberKreaSampler.Resolutions",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CyberKreaEmptyLatent") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const sizeWidget = this.widgets?.find((widget) => widget.name === "size");
            const resolutionWidget = this.widgets?.find(
                (widget) => widget.name === "resolution"
            );
            if (!sizeWidget || !resolutionWidget) return result;

            const updateResolutionOptions = (size) => {
                const resolutions = RESOLUTIONS[size] || [];
                resolutionWidget.options.values = resolutions;
                if (!resolutions.includes(resolutionWidget.value)) {
                    resolutionWidget.value = resolutions[0];
                }
                this.setDirtyCanvas(true, true);
            };

            const originalCallback = sizeWidget.callback;
            sizeWidget.callback = function (value) {
                updateResolutionOptions(value);
                return originalCallback?.apply(this, arguments);
            };
            updateResolutionOptions(sizeWidget.value);
            return result;
        };
    },
});
