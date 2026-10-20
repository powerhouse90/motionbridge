import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
  name: "MotionBridge.SceneUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["MotionBridgeRenderStartingViews", "MotionBridgeLoadCharacter"].includes(nodeData.name)) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      const upload = async (file) => {
        if (!/\.(blend|glb|gltf|fbx)$/i.test(file.name)) throw new Error("Choose a BLEND, GLB, GLTF, or FBX file.");
        const form = new FormData();
        form.append("image", file); form.append("type", "input"); form.append("subfolder", "motionbridge");
        const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
        if (!response.ok) throw new Error(`Upload failed (${response.status})`);
        const data = await response.json();
        this.widgets.find(w => w.name === "asset_file").value = [data.subfolder, data.name].filter(Boolean).join("/");
        app.graph.setDirtyCanvas(true, true);
      };
      const uploadButton = this.addWidget("button", "Upload character / scene", null, () => {
        const input = document.createElement("input"); input.type = "file"; input.accept = ".blend,.glb,.gltf,.fbx";
        input.onchange = () => { if (input.files?.[0]) upload(input.files[0]).catch(e => app.extensionManager.toast.add({ severity: "error", summary: "MotionBridge", detail: e.message })); };
        input.click();
      }, { serialize: false });
      uploadButton.serialize = false;
      const previousDrop = this.onDragDrop;
      this.onDragDrop = function (event) {
        const file = [...(event.dataTransfer?.files || [])].find(f => /\.(blend|glb|gltf|fbx)$/i.test(f.name));
        if (file) { upload(file).catch(e => app.extensionManager.toast.add({ severity: "error", summary: "MotionBridge", detail: e.message })); return true; }
        return previousDrop?.call(this, event) ?? false;
      };
    };
  }
});
