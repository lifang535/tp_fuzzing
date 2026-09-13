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


def _load_runtime_sources():
    # Snapshot at import time so editing source files during a running campaign
    # cannot mix code objects with inspect's newer source line offsets.
    import inspect
    from .runtime import _finite_compare, _max_diff, _dynamic_reference
    return {fn.__name__: inspect.getsource(fn) for fn in
            (_finite_compare, _max_diff, _dynamic_reference)}


_RUNTIME_SOURCES = _load_runtime_sources()


def _threshold_header(config, dynamic=False) -> str:
    """Embed standalone numeric checks and a reproducible input seed."""
    if config is None:
        from src.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    thresholds = {
        "gemm_fp16": config.gemm_rtol_fp16, "gemm_fp32": config.gemm_rtol_fp32,
        "reduce": config.reduce_rtol, "softmax": config.softmax_atol,
        "copy": config.copy_atol, "transpose": config.transpose_atol,
        "elemwise": config.elemwise_atol,
        "pipeline_fp16": config.pipeline_rtol_fp16,
        "pipeline_fp32": config.pipeline_rtol_fp32,
    }
    helpers = ("_finite_compare", "_max_diff")
    if dynamic:
        helpers += ("_dynamic_reference",)
    return (
        f"_THRESHOLDS = {thresholds!r}\n"
        f"torch.manual_seed({config.input_seed!r})\n\n"
        + "\n".join(_RUNTIME_SOURCES[name] for name in helpers)
    )
