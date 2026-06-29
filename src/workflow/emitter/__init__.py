from .tilelang.emitter import TileLangEmitter
from .triton.emitter import TritonEmitter
from .tilelang.pipeline_emitter import TileLangPipelineEmitter
from .triton.pipeline_emitter import TritonPipelineEmitter
from .tilelang.dynamic_emitter import TileLangDynamicEmitter
from .triton.dynamic_emitter import TritonDynamicEmitter


def get_emitter(backend: str, config=None):
    """Return the appropriate emitter for a backend, optionally with config."""
    if backend == "tilelang":
        return TileLangEmitter(config=config)
    elif backend == "triton":
        return TritonEmitter(config=config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _threshold_header(config) -> str:
    """
    Generate a _THRESHOLDS dict and _finite_compare helper for injection at
    the top of every emitted file.

    _finite_compare(C, ref, rtol=None, atol=None):
      Only compares elements where BOTH C and ref are finite (no inf/nan).
      This correctly handles exp() overflow — if ref overflows to inf,
      we skip that element instead of treating it as a bug.
      Returns (max_diff, ref_norm, relative_err) over finite elements only.
    """
    if config is None:
        from src.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    header = (
        f"# Correctness thresholds — set by TileSmith config\n"
        f"_THRESHOLDS = {{\n"
        f"    'gemm_fp16':      {config.gemm_rtol_fp16},\n"
        f"    'gemm_fp32':      {config.gemm_rtol_fp32},\n"
        f"    'reduce':         {config.reduce_rtol},\n"
        f"    'softmax':        {config.softmax_atol},\n"
        f"    'copy':           {config.copy_atol},\n"
        f"    'transpose':      {config.transpose_atol},\n"
        f"    'elemwise':       {config.elemwise_atol},\n"
        f"    'pipeline_fp16':  {config.pipeline_rtol_fp16},\n"
        f"    'pipeline_fp32':  {config.pipeline_rtol_fp32},\n"
        f"}}\n"
        f"\n"
        f"def _finite_compare(C, ref):\n"
        f"    \"\"\"Compare only finite elements. Skip inf/nan overflow cases.\"\"\"\n"
        f"    import torch\n"
        f"    c_f32 = C.to(torch.float32)\n"
        f"    r_f32 = ref.to(torch.float32)\n"
        f"    mask = c_f32.isfinite() & r_f32.isfinite()\n"
        f"    if not mask.any():\n"
        f"        return 0.0, 1.0, 0.0  # all overflow → skip\n"
        f"    diff = (c_f32[mask] - r_f32[mask]).abs()\n"
        f"    max_diff = diff.max().item()\n"
        f"    ref_norm = r_f32[mask].abs().mean().item() + 1e-6\n"
        f"    relative_err = max_diff / ref_norm\n"
        f"    return max_diff, ref_norm, relative_err\n"
    )
    return header
