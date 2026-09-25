import { app } from "../../scripts/app.js";

function aspectOf(resolution) {
    return resolution?.match(/\(([^()]+)\)\s*$/)?.[1];
}

function syncResolutionWidgets(node, resolutionsBySize, selectedSize) {
    const sizeWidget = node.widgets?.find((widget) => widget.name === "size");
    const resolutionWidget = node.widgets?.find(
        (widget) => widget.name === "resolution"
    );
    if (!sizeWidget || !resolutionWidget) return;

    const resolutions = resolutionsBySize[selectedSize ?? sizeWidget.value] || [];
    const previousAspect = aspectOf(resolutionWidget.value);
    resolutionWidget.options.values = resolutions;

    if (!resolutions.includes(resolutionWidget.value)) {
        resolutionWidget.value = resolutions.find(
            (resolution) => aspectOf(resolution) === previousAspect
        ) || resolutions[0];
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "CyberKreaSampler.Resolutions",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CyberKreaEmptyLatent") return;
        const resolutionsBySize = nodeData.input?.required?.size?.[1]?.cyberkrea_resolutions;
        if (!resolutionsBySize) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const sizeWidget = this.widgets?.find((widget) => widget.name === "size");
            const resolutionWidget = this.widgets?.find(
                (widget) => widget.name === "resolution"
            );
            if (!sizeWidget || !resolutionWidget) return result;

            const originalCallback = sizeWidget.callback;
            const node = this;
            sizeWidget.callback = function (value) {
                syncResolutionWidgets(node, resolutionsBySize, value);
                return originalCallback?.apply(this, arguments);
            };
            syncResolutionWidgets(this, resolutionsBySize);
            setTimeout(() => syncResolutionWidgets(this, resolutionsBySize), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => syncResolutionWidgets(this, resolutionsBySize), 0);
            return result;
        };
    },
});
