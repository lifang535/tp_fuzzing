import tilelang
import tilelang.language as T
import json,traceback
dtype="float16"


@tilelang.jit
def typed_kernel_0():
    @T.macro
    def fn_0(arg0, A, B, scratch_0, by, bx, iv, fn_out):
        v4_stat = T.alloc_fragment((64,), "float32")
        v4_wide = T.alloc_fragment((64, 32), "float32")
        v9 = T.alloc_fragment((64, 32), "float32")
        v1 = T.alloc_fragment((64, 32), "float32")
        v2 = T.alloc_fragment((64, 32), "float16")
        v3 = T.alloc_fragment((64, 32), "float16")
        v4 = T.alloc_fragment((64, 1), "float32")
        v5 = T.alloc_fragment((64, 1), "float32")
        v6 = T.alloc_fragment((64, 1), "float32")
        v7 = T.alloc_fragment((64, 32), "float32")
        v8 = T.alloc_fragment((64, 32), "float32")
        v10 = T.alloc_fragment((64, 32), "float32")
        v12 = T.alloc_fragment((64, 32), "float32")
        T.copy(arg0, v9)
        for iter_v9 in T.serial(3):
            T.copy(v9, v1)
            for i, j in T.Parallel(64, 32):
                v2[i, j] = T.if_then_else((by * 64 + i + 1) < 2624 and (bx * 32 + j + 3) < 12025, T.cast(B[by * 64 + i + 1, bx * 32 + j + 3], "float16"), T.cast(0, "float16"))
            for i, j in T.Parallel(64, 32):
                v3[i, j] = T.cast(T.abs(T.cast(v2[i, j], "float32")), "float16")
            for i, j in T.Parallel(64, 32):
                v4_wide[i, j] = T.cast(v3[i, j], "float32")
            T.reduce_min(v4_wide, v4_stat, dim=1, clear=True)
            for i, j in T.Parallel(64, 1):
                v4[i, j] = T.cast(v4_stat[i], "float32")
            for i, j in T.Parallel(64, 1):
                v5[i, j] = T.cast(T.floor(T.cast(v4[i, 0], "float32")), "float32")
            for i, j in T.Parallel(64, 1):
                v6[i, j] = T.cast(T.sin(T.cast(v5[i, 0], "float32")), "float32")
            for i, j in T.Parallel(64, 32):
                v7[i, j] = T.cast(T.cast(v6[i, 0], "float32"), "float32")
            for i, j in T.Parallel(64, 32):
                v8[i, j] = T.cast(T.cast(v7[i, j], "float32") + T.cast(v1[i, j], "float32"), "float32")
            T.copy(v8, v9)
        for i, j in T.Parallel(64, 32):
            v10[i, j] = T.cast(-T.cast(v9[i, j], "float32"), "float32")
        T.sync_threads()
        for i, j in T.Parallel(64, 32):
            scratch_0[(by * 376 + bx) * 2080 + 16 + i * 32 + j] = v10[i, j]
        T.sync_threads()
        for i, j in T.Parallel(64, 32):
            v12[i, j] = T.cast(T.cast(v10[i, j], "float32"), "float32")
        T.copy(v12, fn_out)
    @T.prim_func
    def impl(A: T.Buffer((1241, 2624), dtype), B: T.Buffer((2624, 12025), dtype), scratch_0: T.Buffer((15641600,), "float32"), C: T.Buffer((1241, 12025), dtype)):
        with T.Kernel(20, 376, threads=128) as (by, bx):
            v1 = T.alloc_fragment((64, 32), "float32")
            v2 = T.alloc_fragment((64, 32), "float32")
            v3 = T.alloc_fragment((64, 32), "float32")
            v4 = T.alloc_fragment((64, 32), "float16")
            v5 = T.alloc_fragment((64, 32), "float32")
            v6 = T.alloc_fragment((64, 32), "float32")
            v7 = T.alloc_fragment((64, 32), "float32")
            As = T.alloc_shared((64, 32), dtype)
            Bs = T.alloc_shared((32, 32), dtype)
            T.clear(v1)
            for ki in T.Pipelined(82, num_stages=4):
                T.copy(A[by * 64, ki * 32], As)
                T.copy(B[ki * 32, bx * 32], Bs)
                T.gemm(As, Bs, v1)
            for i, j in T.Parallel(64, 32):
                v2[i, j] = T.cast(T.cast(v1[i, j], "float32") + T.cast(by, "float32") * 0.5, "float32")
            for i, j in T.Parallel(64, 32):
                v3[i, j] = T.cast(T.log2(T.max(T.abs(T.cast(v2[i, j], "float32")), T.float32(0.001))), "float32")
            for i, j in T.Parallel(64, 32):
                v4[i, j] = T.cast(T.cast(v3[i, j], "float32"), "float16")
            for i, j in T.Parallel(64, 32):
                v5[i, j] = T.cast(T.cast(v4[i, j], "float32"), "float32")
            for i, j in T.Parallel(64, 32):
                v6[i, j] = T.cast(T.cast(v5[i, j], "float32"), "float32")
            fn_0(v5, A, B, scratch_0, by, bx, 0, v7)
            for i, j in T.Parallel(64, 32):
                if by*64+i < 1241 and bx*32+j < 12025:
                    C[by*64+i, bx*32+j] = v7[i, j]
    return impl


for name in ['typed_kernel_0']:
 for option,config in [('default',{}),('no_vector',{'tirx.disable_vectorize':True}),('no_async',{'tl.enable_async_copy':False})]:
  try:
   ir=globals()[name].get_tir();tilelang.compile(ir,target='cuda',pass_configs=config)
   print(json.dumps({'kernel':name,'option':option,'result':'PASS'}),flush=True)
  except Exception as e:
   print(json.dumps({'kernel':name,'option':option,'result':'FAIL','error':str(e)[-1800:]}),flush=True)
