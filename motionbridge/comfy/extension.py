from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from motionbridge.comfy import nodes, references


class MotionBridgeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [references.RenderStartingViews, references.AssembleReferenceViews] + nodes.NODES


async def comfy_entrypoint() -> MotionBridgeExtension:
    return MotionBridgeExtension()
