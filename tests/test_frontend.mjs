import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const metadata = JSON.parse(readFileSync(0, "utf8"));

function loadExtension(filename) {
    let extension;
    const timers = [];
    const source = readFileSync(new URL(`../web/${filename}`, import.meta.url), "utf8")
        .replace(/^import .*;\s*/m, "");
    vm.runInNewContext(source, {
        app: { registerExtension(value) { extension = value; } },
        setTimeout(callback) { timers.push(callback); },
    });
    return { extension, flush() { while (timers.length) timers.shift()(); } };
}

function makeNodeType(definition) {
    return class Node {
        constructor() {
            this.widgets = Object.entries(definition.input.required).map(([name, [type, options]]) => ({
                name, value: options?.default ?? (Array.isArray(type) ? type[0] : undefined),
                options: { values: Array.isArray(type) ? [...type] : undefined },
            }));
            this.events = [];
        }
        onNodeCreated() { this.events.push("created"); return "created-result"; }
        onConfigure() { this.events.push("configured"); return "configured-result"; }
        setDirtyCanvas() { this.dirty = true; }
        widget(name) { return this.widgets.find(widget => widget.name === name); }
    };
}

// A changed backend value must reach the widgets without any frontend table edit.
const samplerData = structuredClone(metadata.sampler);
samplerData.input.required.preset[1].cyberkrea_presets.fast.steps = 7;
const sampler = loadExtension("cyberkrea_presets.js");
const Sampler = makeNodeType(samplerData);
await sampler.extension.beforeRegisterNodeDef(Sampler, samplerData);
const node = new Sampler();
let callbackReceiver;
node.widget("preset").callback = function () { callbackReceiver = this; return "callback-result"; };
assert.equal(node.onNodeCreated(), "created-result");
assert.equal(node.widget("steps").value, 12);
for (const [name, values] of Object.entries(samplerData.input.required.preset[1].cyberkrea_presets)) {
    node.widget("preset").value = name;
    assert.equal(node.widget("preset").callback(name), "callback-result");
    assert.equal(callbackReceiver, node.widget("preset"));
    for (const [key, value] of Object.entries(values)) assert.equal(node.widget(key).value, value);
}
node.widget("steps").value = 23;
assert.equal(node.onConfigure(), "configured-result");
assert.equal(node.widget("steps").value, 23);
assert.equal(node.widget("preview_method").value, "default");

const resolutions = loadExtension("cyberkrea_resolutions.js");
const ResolutionNode = makeNodeType(metadata.resolutions);
await resolutions.extension.beforeRegisterNodeDef(ResolutionNode, metadata.resolutions);
const latent = new ResolutionNode();
const size = latent.widget("size");
let sizeCallbackReceiver;
size.callback = function () { sizeCallbackReceiver = this; return "size-result"; };
assert.equal(latent.onNodeCreated(), "created-result");
resolutions.flush();
const resolution = latent.widget("resolution");
const tables = metadata.resolutions.input.required.size[1].cyberkrea_resolutions;
for (const aspect of ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]) {
    resolution.value = tables["L (~1.7 MP)"].find(value => value.endsWith(`(${aspect})`));
    for (const [tier, values] of Object.entries(tables)) {
        size.value = tier;
        assert.equal(size.callback(tier), "size-result");
        assert.equal(sizeCallbackReceiver, size);
        assert.deepEqual([...resolution.options.values], values);
        assert.equal(resolution.value, values.find(value => value.endsWith(`(${aspect})`)));
    }
}
// Restoring a workflow may temporarily leave a resolution from the old tier.
size.value = "XL (~2.1 MP)";
resolution.value = "1600x1088 (3:2)";
assert.equal(latent.onConfigure(), "configured-result");
resolutions.flush();
assert.equal(resolution.value, "1776x1184 (3:2)");
resolution.value = "unknown";
size.callback("S (~1.0 MP)");
assert.equal(resolution.value, tables["S (~1.0 MP)"][0]);

// A different node or an older definition without metadata is left alone.
for (const [extension, name] of [[sampler.extension, "CyberKreaSampler"],
                                [resolutions.extension, "CyberKreaEmptyLatent"]]) {
    const Other = makeNodeType(metadata.sampler);
    const original = Other.prototype.onNodeCreated;
    await extension.beforeRegisterNodeDef(Other, { name: "Other" });
    await extension.beforeRegisterNodeDef(Other, { name });
    assert.equal(Other.prototype.onNodeCreated, original);
}
console.log("Frontend presets, manual overrides, resolution tiers and workflow restoration passed.");
