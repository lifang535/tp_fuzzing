"""
Op Registry — One class per ComputeKind.

Each op class exposes four static methods used by the emitters:
  - tilelang_kernel_body(k, sp) -> list[str]   : kernel body lines (TileLang)
  - tilelang_test_body(k)       -> list[str]   : test function body lines (TileLang)
  - triton_kernel_args(k)       -> str         : @triton.jit argument list
  - triton_kernel_body(k, sp)   -> list[str]   : kernel body lines (Triton)
  - triton_launch_and_test(k)   -> list[str]   : launch + test comparison lines (Triton)

The emitters wrap these with boilerplate (imports, function signatures, etc.).
"""

from src.ir.ir import ComputeKind, TileKernel, LoopKind, DataType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _torch_dtype(dtype: DataType) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype.value, "float16")


def _tl_dtype(dtype: DataType) -> str:
    return {"float16": "tl.float16", "float32": "tl.float32"}.get(dtype.value, "tl.float16")


def _tilelang_loop_stmt(k: TileKernel) -> str:
    if k.loop_kind == LoopKind.PIPELINED:
        return f"for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={k.num_stages}):"
    else:
        return f"for k in T.serial(T.ceildiv(K, block_K)):"


def _triton_loop_stmt(k: TileKernel) -> str:
    if k.loop_kind == LoopKind.PIPELINED:
        return f"for k in tl.range(0, K, BLOCK_K, num_stages={k.num_stages}):"
    else:
        return f"for k in range(0, K, BLOCK_K):"


# ─────────────────────────────────────────────────────────────────────────────
# GEMM
# ─────────────────────────────────────────────────────────────────────────────

class GemmOp:
    kind = ComputeKind.GEMM

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        loop_stmt = _tilelang_loop_stmt(k)
        return [
            f"{sp}A_shared = T.alloc_shared((block_M, block_K), dtype)",
            f"{sp}B_shared = T.alloc_shared((block_K, block_N), dtype)",
            f"{sp}C_local = T.alloc_fragment((block_M, block_N), accum_dtype)",
            f"{sp}T.clear(C_local)",
            f"{sp}{loop_stmt}",
            f"{sp}    T.copy(A[by * block_M, k * block_K], A_shared)",
            f"{sp}    T.copy(B[k * block_K, bx * block_N], B_shared)",
            f"{sp}    T.gemm(A_shared, B_shared, C_local)",
            f"{sp}T.copy(C_local, C[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N, K = {k.M}, {k.N}, {k.K}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    # Reference: GEMM semantics = A @ B",
            f"    ref = A.float() @ B.float()",
            f"    C_f32 = C.to(torch.float32)",
            f"    max_diff = (C_f32 - ref).abs().max().item()",
            f"    ref_norm = ref.abs().mean().item() + 1e-6",
            f"    relative_err = max_diff / ref_norm",
            f'    threshold = _THRESHOLDS["gemm_fp32"] if "{k.dtype.value}" == "float32" else _THRESHOLDS["gemm_fp16"]',
            f"    if relative_err > threshold:",
            f'        raise RuntimeError(f"WRONG RESULT [gemm]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return (
            "    a_ptr, b_ptr, c_ptr, M, N, K,\n"
            "    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,\n"
            "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,"
        )

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        tld = _tl_dtype(k.dtype)
        loop_stmt = _triton_loop_stmt(k)
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}offs_k = tl.arange(0, BLOCK_K)",
            f"{sp}acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
            f"{sp}{loop_stmt}",
            f"{sp}    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)",
            f"{sp}    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0).to({tld})",
            f"{sp}    b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)",
            f"{sp}    b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0).to({tld})",
            f"{sp}    acc += tl.dot(a, b)",
            f"{sp}c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)",
            f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N, K = {k.M}, {k.N}, {k.K}",
            f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')",
            f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](",
            f"        A, B, C, M, N, K,",
            f"        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),",
            f"        BLOCK_M={k.block_M}, BLOCK_N={k.block_N}, BLOCK_K={k.block_K},",
            f"    )",
            f"    return A, B, C",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B, C = {k.name}()",
            f"    ref = A.float() @ B.float()",
            f"    C_f32 = C.to(torch.float32)",
            f"    max_diff = (C_f32 - ref).abs().max().item()",
            f"    ref_norm = ref.abs().mean().item() + 1e-6",
            f"    relative_err = max_diff / ref_norm",
            f'    threshold = _THRESHOLDS["gemm_fp32"] if "{k.dtype.value}" == "float32" else _THRESHOLDS["gemm_fp16"]',
            f"    if relative_err > threshold:",
            f'        raise RuntimeError(f"WRONG RESULT [gemm]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# COPY
# ─────────────────────────────────────────────────────────────────────────────

class CopyOp:
    kind = ComputeKind.COPY

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
            f"{sp}T.copy(A_local, B[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = kernel(A)",
            f"    # Reference: COPY semantics = identity",
            f"    ref = A",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["copy"]:',
            f'        raise RuntimeError(f"WRONG RESULT [copy]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}data = tl.load(ptrs, mask=mask, other=0.0)",
            f"{sp}out_ptrs = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(out_ptrs, data, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    max_diff = (B.to(torch.float32) - A.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["copy"]:',
            f'        raise RuntimeError(f"WRONG RESULT [copy]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Binary elementwise base helper
# ─────────────────────────────────────────────────────────────────────────────

def _tilelang_binary_elemwise_body(k: TileKernel, sp: str, tl_expr: str) -> list:
    return [
        f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
        f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
        f"{sp}C_local = T.alloc_fragment((block_M, block_N), dtype)",
        f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
        f"{sp}T.copy(B[by * block_M, bx * block_N], B_local)",
        f"{sp}for i, j in T.Parallel(block_M, block_N):",
        f"{sp}    C_local[i, j] = {tl_expr}",
        f"{sp}T.copy(C_local, C[by * block_M, bx * block_N])",
    ]


def _triton_binary_elemwise_body(sp: str, expr: str) -> list:
    return [
        f"{sp}pid_m = tl.program_id(0)",
        f"{sp}pid_n = tl.program_id(1)",
        f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        f"{sp}ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
        f"{sp}ptrs_b = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
        f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
        f"{sp}a = tl.load(ptrs_a, mask=mask, other=0.0)",
        f"{sp}b = tl.load(ptrs_b, mask=mask, other=0.0)",
        f"{sp}c = {expr}",
        f"{sp}out_ptrs = c_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
        f"{sp}tl.store(out_ptrs, c, mask=mask)",
    ]


def _triton_binary_launch_test(k: TileKernel, ref_expr: str, op_name: str) -> list:
    td = _torch_dtype(k.dtype)
    return [
        f"def {k.name}():",
        f"    M, N = {k.M}, {k.N}",
        f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
        f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
        f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
        f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
        f"    {k.name}_kernel[grid](A, B, C, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
        f"    return A, B, C",
        f"",
        f"",
        f"def test_{k.name}():",
        f"    A, B, C = {k.name}()",
        f"    ref = {ref_expr}",
        f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
        f'    if max_diff > _THRESHOLDS["elemwise"]:',
        f'        raise RuntimeError(f"WRONG RESULT [{op_name}]: max_diff={{max_diff:.6f}}")',
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ELEMWISE_ADD
# ─────────────────────────────────────────────────────────────────────────────

class ElemwiseAddOp:
    kind = ComputeKind.ELEMWISE_ADD

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_binary_elemwise_body(k, sp, "A_local[i, j] + B_local[i, j]")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    ref = (A + B).to(torch.float32)",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [elemwise_add]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return _triton_binary_elemwise_body(sp, "a + b")

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        return _triton_binary_launch_test(k, "A + B", "elemwise_add")


# ─────────────────────────────────────────────────────────────────────────────
# ELEMWISE_MUL
# ─────────────────────────────────────────────────────────────────────────────

class ElemwiseMulOp:
    kind = ComputeKind.ELEMWISE_MUL

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_binary_elemwise_body(k, sp, "A_local[i, j] * B_local[i, j]")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    ref = (A * B).to(torch.float32)",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [elemwise_mul]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return _triton_binary_elemwise_body(sp, "a * b")

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        return _triton_binary_launch_test(k, "A * B", "elemwise_mul")


# ─────────────────────────────────────────────────────────────────────────────
# ELEMWISE_MAX
# ─────────────────────────────────────────────────────────────────────────────

class ElemwiseMaxOp:
    kind = ComputeKind.ELEMWISE_MAX

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_binary_elemwise_body(
            k, sp,
            "A_local[i, j] if A_local[i, j] > B_local[i, j] else B_local[i, j]",
        )

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    ref = torch.maximum(A, B).to(torch.float32)",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [elemwise_max]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return _triton_binary_elemwise_body(sp, "tl.maximum(a, b)")

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        return _triton_binary_launch_test(k, "torch.maximum(A, B)", "elemwise_max")


# ─────────────────────────────────────────────────────────────────────────────
# ELEMWISE_SUB
# ─────────────────────────────────────────────────────────────────────────────

class ElemwiseSubOp:
    kind = ComputeKind.ELEMWISE_SUB

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_binary_elemwise_body(k, sp, "A_local[i, j] - B_local[i, j]")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    ref = (A - B).to(torch.float32)",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [elemwise_sub]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return _triton_binary_elemwise_body(sp, "a - b")

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        return _triton_binary_launch_test(k, "A - B", "elemwise_sub")


# ─────────────────────────────────────────────────────────────────────────────
# SCALE
# ─────────────────────────────────────────────────────────────────────────────

class ScaleOp:
    kind = ComputeKind.SCALE

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_local[i, j] = A_local[i, j] * {k.alpha}",
            f"{sp}T.copy(B_local, B[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = kernel(A)",
            f"    ref = ({k.alpha} * A.float()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [scale]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, M, N, stride_m, stride_n, alpha, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(ptrs_a, mask=mask, other=0.0)",
            f"{sp}b = a * alpha",
            f"{sp}out_ptrs = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(out_ptrs, b, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), {k.alpha}, BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = ({k.alpha} * A.float()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [scale]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# UNARY_EXP
# ─────────────────────────────────────────────────────────────────────────────

class UnaryExpOp:
    kind = ComputeKind.UNARY_EXP

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_local[i, j] = T.exp(A_local[i, j])",
            f"{sp}T.copy(B_local, B[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda').clamp(-10, 10)",
            f"    B = kernel(A)",
            f"    ref = torch.exp(A.float()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [unary_exp]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        # tl.exp only supports fp32/fp64 — cast to fp32, compute, cast back
        tld = _tl_dtype(k.dtype)
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(ptrs_a, mask=mask, other=0.0).to(tl.float32)",
            f"{sp}b = tl.exp(a).to({tld})",
            f"{sp}out_ptrs = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(out_ptrs, b, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda').clamp(-10, 10)",
            f"    B = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = torch.exp(A.float()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [unary_exp]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# UNARY_SQRT
# ─────────────────────────────────────────────────────────────────────────────

class UnarySqrtOp:
    kind = ComputeKind.UNARY_SQRT

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_local[i, j] = T.sqrt(T.abs(A_local[i, j]))",
            f"{sp}T.copy(B_local, B[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = kernel(A)",
            f"    ref = torch.sqrt(A.float().abs()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [unary_sqrt]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(ptrs_a, mask=mask, other=0.0)",
            f"{sp}b = tl.sqrt(tl.abs(a.to(tl.float32)))",
            f"{sp}out_ptrs = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(out_ptrs, b, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = torch.sqrt(A.float().abs()).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [unary_sqrt]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSPOSE
# ─────────────────────────────────────────────────────────────────────────────

class TransposeOp:
    kind = ComputeKind.TRANSPOSE

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        # B shape is (N, M); grid is (ceildiv(M, block_M), ceildiv(N, block_N))
        # bx -> M-axis block, by -> N-axis block
        return [
            f"{sp}A_shared = T.alloc_shared((block_M, block_N), dtype)",
            f"{sp}B_shared = T.alloc_shared((block_N, block_M), dtype)",
            f"{sp}T.copy(A[bx * block_M, by * block_N], A_shared)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_shared[j, i] = A_shared[i, j]",
            f"{sp}T.copy(B_shared, B[by * block_N, bx * block_M])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = kernel(A)",
            f"    ref = A.T.contiguous()",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["transpose"]:',
            f'        raise RuntimeError(f"WRONG RESULT [transpose]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return (
            "    a_ptr, b_ptr, M, N,\n"
            "    stride_am, stride_an, stride_bm, stride_bn,\n"
            "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"
        )

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        # pid_m iterates over M (rows of A), pid_n over N (cols of A)
        # B[j, i] = A[i, j] => B has shape (N, M)
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}# Load A tile: shape (BLOCK_M, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(a_ptrs, mask=mask, other=0.0)",
            f"{sp}# Transpose: write B[offs_n, offs_m]",
            f"{sp}b_ptrs = b_ptr + offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn",
            f"{sp}t_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)",
            f"{sp}tl.store(b_ptrs, tl.trans(a), mask=t_mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(N, M, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](",
            f"        A, B, M, N,",
            f"        A.stride(0), A.stride(1), B.stride(0), B.stride(1),",
            f"        BLOCK_M={k.block_M}, BLOCK_N={k.block_N},",
            f"    )",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = A.T.contiguous()",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["transpose"]:',
            f'        raise RuntimeError(f"WRONG RESULT [transpose]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Reduce ops base helper
# ─────────────────────────────────────────────────────────────────────────────

def _tilelang_reduce_body(k: TileKernel, sp: str, reduce_fn: str) -> list:
    return [
        f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
        f"{sp}B_local = T.alloc_fragment((block_M,), dtype)",
        f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
        f"{sp}{reduce_fn}(A_local, B_local, dim=1, clear=True)",
        f"{sp}T.copy(B_local, B[by * block_M])",
    ]


def _triton_reduce_body(sp: str, reduce_fn: str) -> list:
    return [
        f"{sp}pid_m = tl.program_id(0)",
        f"{sp}pid_n = tl.program_id(1)",
        f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
        f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
        f"{sp}a = tl.load(a_ptrs, mask=mask, other=0.0)",
        f"{sp}b = {reduce_fn}(a, axis=1)",
        f"{sp}b_ptrs = b_ptr + offs_m * stride_bm",
        f"{sp}b_mask = offs_m < M",
        f"{sp}tl.atomic_add(b_ptrs, b, mask=b_mask)" if reduce_fn == "tl.sum" else
        f"{sp}tl.store(b_ptrs, b, mask=b_mask)",
    ]


def _tilelang_reduce_test(k: TileKernel, ref_expr: str, op_name: str) -> list:
    td = _torch_dtype(k.dtype)
    return [
        f"    M, N = {k.M}, {k.N}",
        f"    kernel = {k.name}()",
        f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
        f"    B = kernel(A)",
        f"    ref = {ref_expr}",
        f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
        f"    ref_norm = ref.float().abs().mean().item() + 1e-6",
        f"    relative_err = max_diff / ref_norm",
        f'    if relative_err > _THRESHOLDS["reduce"]:',
        f'        raise RuntimeError(f"WRONG RESULT [{op_name}]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")',
    ]


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE_SUM
# ─────────────────────────────────────────────────────────────────────────────

class ReduceSumOp:
    kind = ComputeKind.REDUCE_SUM

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_reduce_body(k, sp, "T.reduce_sum")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return _tilelang_reduce_test(
            k,
            f"A.float().sum(dim=1).to(torch.{td})",
            "reduce_sum",
        )

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return (
            "    a_ptr, b_ptr, M, N, stride_m, stride_n, stride_bm,"
            " BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"
        )

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(a_ptrs, mask=mask, other=0.0)",
            f"{sp}b = tl.sum(a, axis=1)",
            f"{sp}b_ptrs = b_ptr + offs_m * stride_bm",
            f"{sp}b_mask = offs_m < M",
            f"{sp}tl.atomic_add(b_ptrs, b, mask=b_mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.zeros(M, dtype=torch.float32, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), B.stride(0), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = A.float().sum(dim=1)",
            f"    max_diff = (B - ref).abs().max().item()",
            f"    ref_norm = ref.abs().mean().item() + 1e-6",
            f"    relative_err = max_diff / ref_norm",
            f'    if relative_err > _THRESHOLDS["reduce"]:',
            f'        raise RuntimeError(f"WRONG RESULT [reduce_sum]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE_MAX
# ─────────────────────────────────────────────────────────────────────────────

class ReduceMaxOp:
    kind = ComputeKind.REDUCE_MAX

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_reduce_body(k, sp, "T.reduce_max")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return _tilelang_reduce_test(
            k,
            f"A.float().max(dim=1).values.to(torch.{td})",
            "reduce_max",
        )

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return (
            "    a_ptr, b_ptr, M, N, stride_m, stride_n, stride_bm,"
            " BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"
        )

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(a_ptrs, mask=mask, other=float('-inf'))",
            f"{sp}b = tl.max(a, axis=1)",
            f"{sp}b_ptrs = b_ptr + offs_m * stride_bm",
            f"{sp}b_mask = offs_m < M",
            f"{sp}tl.store(b_ptrs, b, mask=b_mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), B.stride(0), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = A.float().max(dim=1).values",
            f"    max_diff = (B.to(torch.float32) - ref).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["reduce"]:',
            f'        raise RuntimeError(f"WRONG RESULT [reduce_max]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE_MIN
# ─────────────────────────────────────────────────────────────────────────────

class ReduceMinOp:
    kind = ComputeKind.REDUCE_MIN

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return _tilelang_reduce_body(k, sp, "T.reduce_min")

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return _tilelang_reduce_test(
            k,
            f"A.float().min(dim=1).values.to(torch.{td})",
            "reduce_min",
        )

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return (
            "    a_ptr, b_ptr, M, N, stride_m, stride_n, stride_bm,"
            " BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"
        )

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(a_ptrs, mask=mask, other=float('inf'))",
            f"{sp}b = tl.min(a, axis=1)",
            f"{sp}b_ptrs = b_ptr + offs_m * stride_bm",
            f"{sp}b_mask = offs_m < M",
            f"{sp}tl.store(b_ptrs, b, mask=b_mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), B.stride(0), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = A.float().min(dim=1).values",
            f"    max_diff = (B.to(torch.float32) - ref).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["reduce"]:',
            f'        raise RuntimeError(f"WRONG RESULT [reduce_min]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SOFTMAX
# ─────────────────────────────────────────────────────────────────────────────

class SoftmaxOp:
    kind = ComputeKind.SOFTMAX

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        # For softmax: N == block_N (full row at once), so bx is always 0
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}max_local = T.alloc_fragment((block_M,), dtype)",
            f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}sum_local = T.alloc_fragment((block_M,), dtype)",
            f"{sp}T.copy(A[by * block_M, 0], A_local)",
            f"{sp}T.reduce_max(A_local, max_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_local[i, j] = T.exp(A_local[i, j] - max_local[i])",
            f"{sp}T.reduce_sum(B_local, sum_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    B_local[i, j] = B_local[i, j] / sum_local[i]",
            f"{sp}T.copy(B_local, B[by * block_M, 0])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = kernel(A)",
            f"    ref = torch.softmax(A.float(), dim=-1).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["softmax"]:',
            f'        raise RuntimeError(f"WRONG RESULT [softmax]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = tl.arange(0, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(a_ptrs, mask=mask, other=float('-inf'))",
            f"{sp}b = tl.softmax(a, 1)",  # axis keyword removed in Triton >= 2.2
            f"{sp}b_ptrs = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(b_ptrs, b, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        # Triton softmax kernel uses 1D grid (one block per row group)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}),)",
            f"    {k.name}_kernel[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B = {k.name}()",
            f"    ref = torch.softmax(A.float(), dim=-1).to(torch.{td})",
            f"    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["softmax"]:',
            f'        raise RuntimeError(f"WRONG RESULT [softmax]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# WHERE
# ─────────────────────────────────────────────────────────────────────────────

class WhereOp:
    kind = ComputeKind.WHERE

    @staticmethod
    def tilelang_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}A_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}B_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}C_local = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy(A[by * block_M, bx * block_N], A_local)",
            f"{sp}T.copy(B[by * block_M, bx * block_N], B_local)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    C_local[i, j] = A_local[i, j] if A_local[i, j] > 0.0 else B_local[i, j]",
            f"{sp}T.copy(C_local, C[by * block_M, bx * block_N])",
        ]

    @staticmethod
    def tilelang_test_body(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"    M, N = {k.M}, {k.N}",
            f"    kernel = {k.name}()",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = kernel(A, B)",
            f"    ref = torch.where(A > 0, A, B).to(torch.{td})",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [where]: max_diff={{max_diff:.6f}}")',
        ]

    @staticmethod
    def triton_kernel_args(k: TileKernel) -> str:
        return "    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"

    @staticmethod
    def triton_kernel_body(k: TileKernel, sp: str) -> list:
        return [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}ptrs_b = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}a = tl.load(ptrs_a, mask=mask, other=0.0)",
            f"{sp}b = tl.load(ptrs_b, mask=mask, other=0.0)",
            f"{sp}c = tl.where(a > 0, a, b)",
            f"{sp}out_ptrs = c_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n",
            f"{sp}tl.store(out_ptrs, c, mask=mask)",
        ]

    @staticmethod
    def triton_launch_and_test(k: TileKernel) -> list:
        td = _torch_dtype(k.dtype)
        return [
            f"def {k.name}():",
            f"    M, N = {k.M}, {k.N}",
            f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    B = torch.randn(M, N, dtype=torch.{td}, device='cuda')",
            f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')",
            f"    grid = (triton.cdiv(M, {k.block_M}), triton.cdiv(N, {k.block_N}))",
            f"    {k.name}_kernel[grid](A, B, C, M, N, A.stride(0), A.stride(1), BLOCK_M={k.block_M}, BLOCK_N={k.block_N})",
            f"    return A, B, C",
            f"",
            f"",
            f"def test_{k.name}():",
            f"    A, B, C = {k.name}()",
            f"    ref = torch.where(A > 0, A, B).to(torch.{td})",
            f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()",
            f'    if max_diff > _THRESHOLDS["elemwise"]:',
            f'        raise RuntimeError(f"WRONG RESULT [where]: max_diff={{max_diff:.6f}}")',
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

OP_REGISTRY = {
    ComputeKind.GEMM:        GemmOp,
    ComputeKind.COPY:        CopyOp,
    ComputeKind.ELEMWISE_ADD: ElemwiseAddOp,
    ComputeKind.ELEMWISE_MUL: ElemwiseMulOp,
    ComputeKind.ELEMWISE_MAX: ElemwiseMaxOp,
    ComputeKind.ELEMWISE_SUB: ElemwiseSubOp,
    ComputeKind.SCALE:       ScaleOp,
    ComputeKind.UNARY_EXP:   UnaryExpOp,
    ComputeKind.UNARY_SQRT:  UnarySqrtOp,
    ComputeKind.TRANSPOSE:   TransposeOp,
    ComputeKind.REDUCE_SUM:  ReduceSumOp,
    ComputeKind.REDUCE_MAX:  ReduceMaxOp,
    ComputeKind.REDUCE_MIN:  ReduceMinOp,
    ComputeKind.SOFTMAX:     SoftmaxOp,
    ComputeKind.WHERE:       WhereOp,
}
