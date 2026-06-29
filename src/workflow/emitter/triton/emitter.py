"""
Triton Code Emitter — Translates TileProgram IR to Triton executable code.
"""

from src.ir.ir import TileProgram, TileKernel, ComputeKind, LoopKind, DataType, REDUCE_OPS
from src.ops import OP_REGISTRY
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: DataType) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype.value, "float16")


class TritonEmitter:
    """Emits Triton code. Same IR → same semantics, different syntax."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, program: TileProgram) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import triton",
            "import triton.language as tl",
            "import torch",
            "",
            _threshold_header(self.config),
            "",
        ]
        for kernel in program.kernels:
            lines.append(self._emit_kernel(kernel))
            lines.append("")

        lines.append("if __name__ == '__main__':")
        for kernel in program.kernels:
            lines.append(f"    test_{kernel.name}()")
            lines.append(f"    print('{kernel.name} PASSED')")
        lines.append("    print('ALL PASSED')")
        return "\n".join(lines)

    def _emit_kernel(self, k: TileKernel) -> str:
        op_cls = OP_REGISTRY.get(k.compute_kind)
        if op_cls is None:
            raise ValueError(f"No op class registered for {k.compute_kind}")

        sp = "    "  # indentation inside @triton.jit function
        kernel_args = op_cls.triton_kernel_args(k)
        kernel_body = op_cls.triton_kernel_body(k, sp)
        launch_test = op_cls.triton_launch_and_test(k)

        body_str = "\n".join(kernel_body)
        launch_str = "\n".join(launch_test)

        return (
            f"@triton.jit\n"
            f"def {k.name}_kernel(\n"
            f"{kernel_args}\n"
            f"):\n"
            f"{body_str}\n"
            f"\n"
            f"\n"
            f"{launch_str}"
        )
