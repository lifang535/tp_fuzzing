import tilelang
import tilelang.language as T
import json,traceback
dtype="float16"


@tilelang.jit
def typed_kernel_0():
    @T.macro
    def fn_0(arg0, A, B, scratch_0, by, bx, iv, fn_out):
        v3_stat = T.alloc_fragment((64,), "float32")
        v3_wide = T.alloc_fragment((64, 64), "float32")
        v1 = T.alloc_fragment((64, 64), "float32")
        v8 = T.alloc_fragment((64, 64), "float32")
        v2 = T.alloc_fragment((64, 64), "float32")
        v5 = T.alloc_fragment((64, 64), "float32")
        v3 = T.alloc_fragment((64, 64), "float32")
        v4 = T.alloc_fragment((64, 64), "float32")
        v6 = T.alloc_fragment((64, 64), "float32")
        v7 = T.alloc_fragment((64, 64), "float32")
        v9 = T.alloc_fragment((64, 64), "float32")
        for i, j in T.Parallel(64, 64):
            v1[i, j] = T.cast(T.tanh(T.cast(arg0[i, j], "float32")), "float32")
        if bx % 4 == 2:
            T.copy(v1, v2)
            for i, j in T.Parallel(64, 64):
                v3_wide[i, j] = T.cast(v1[i, j], "float32")
            T.reduce_sum(v3_wide, v3_stat, dim=1, clear=True)
            for i, j in T.Parallel(64, 64):
                v3[i, j] = T.cast(v3_stat[i], "float32")
            for i, j in T.Parallel(64, 64):
                v4[i, j] = T.cast(T.cast(v3[i, j], "float32"), "float32")
            T.copy(v4, v8)
        else:
            T.copy(v1, v5)
            for i, j in T.Parallel(64, 64):
                v6[i, j] = T.cast(T.cast(v5[i, j], "float32") / T.max(T.abs(T.cast(arg0[i, j], "float32")), T.float32(0.001)), "float32")
            for i, j in T.Parallel(64, 64):
                v7[i, j] = T.cast(T.cast(v6[i, j], "float32"), "float32")
            T.copy(v7, v8)
        for i, j in T.Parallel(64, 64):
            v9[i, j] = T.cast(T.cast(v1[i, j], "float32"), "float32")
        T.copy(v9, fn_out)
    @T.macro
    def fn_1(arg0, A, B, scratch_0, by, bx, iv, fn_out):
        v1 = T.alloc_fragment((64, 64), "float32")
        v2 = T.alloc_fragment((64, 64), "float32")
        v3 = T.alloc_fragment((64, 64), "float32")
        v4 = T.alloc_fragment((64, 64), "float32")
        v6 = T.alloc_fragment((64, 64), "float32")
        v7 = T.alloc_fragment((64, 64), "float32")
        for i, j in T.Parallel(64, 64):
            v1[i, j] = T.cast(T.sin(T.cast(arg0[i, j], "float32")), "float32")
        for i, j in T.Parallel(64, 64):
            v2[i, j] = T.cast(T.exp2(T.min(T.max(T.cast(v1[i, j], "float32"), T.float32(-10)), T.float32(10))), "float32")
        for i, j in T.Parallel(64, 64):
            v3[i, j] = T.cast(T.cast(v2[i, j], "float32") + T.cast(bx, "float32") * 0.25, "float32")
        for i, j in T.Parallel(64, 64):
            v4[i, j] = T.cast(T.erf(T.cast(v3[i, j], "float32")), "float32")
        T.sync_threads()
        for i, j in T.Parallel(64, 64):
            scratch_0[(by * 58 + bx) * 4128 + 16 + i * 64 + j] = v4[i, j]
        T.sync_threads()
        for i, j in T.Parallel(64, 64):
            v6[i, j] = T.cast(T.cast(v1[i, j], "float32"), "float32")
        fn_0(v6, A, B, scratch_0, by, bx, iv, v7)
        T.copy(v7, fn_out)
    @T.prim_func
    def impl(A: T.Buffer((1, 9157), dtype), B: T.Buffer((9157, 3709), dtype), scratch_0: T.Buffer((239424,), "float32"), C: T.Buffer((1, 3709), dtype)):
        with T.Kernel(1, 58, threads=256) as (by, bx):
            v1 = T.alloc_fragment((64, 64), "float32")
            v2 = T.alloc_fragment((64, 64), "float16")
            v3 = T.alloc_fragment((64, 64), "float16")
            v8 = T.alloc_fragment((64, 64), "float32")
            v4 = T.alloc_fragment((64, 64), "float32")
            v5 = T.alloc_fragment((64, 64), "float32")
            v6 = T.alloc_fragment((64, 64), "float32")
            v7 = T.alloc_fragment((64, 64), "float32")
            v9 = T.alloc_fragment((64, 64), "float32")
            v10 = T.alloc_fragment((64, 64), "float32")
            As = T.alloc_shared((64, 128), dtype)
            Bs = T.alloc_shared((128, 64), dtype)
            T.clear(v1)
            for ki in T.Pipelined(72, num_stages=4):
                T.copy(A[by * 64, ki * 128], As)
                T.copy(B[ki * 128, bx * 64], Bs)
                T.gemm(As, Bs, v1)
            for i, j in T.Parallel(64, 64):
                v2[i, j] = T.if_then_else((by * 64 + i + 0) < 1 and (bx * 64 + j + 1) < 9157, T.cast(A[by * 64 + i + 0, bx * 64 + j + 1], "float16"), T.cast(0, "float16"))
            for i, j in T.Parallel(64, 64):
                v3[i, j] = T.cast(T.cast(v2[i, j], "float32") + T.cast(by, "float32") * 0.5, "float16")
            T.copy(v1, v8)
            for iter_v8 in T.serial(3):
                T.copy(v8, v4)
                for i, j in T.Parallel(64, 64):
                    v5[i, j] = T.cast(T.log2(T.max(T.abs(T.cast(v4[i, j], "float32")), T.float32(0.001))), "float32")
                for i, j in T.Parallel(64, 64):
                    v6[i, j] = T.cast(T.cast(v5[i, j], "float32"), "float32")
                for i, j in T.Parallel(64, 64):
                    v7[i, j] = T.cast(T.cast(v6[i, j], "float32") + T.cast(v4[i, j], "float32"), "float32")
                T.copy(v7, v8)
            for i, j in T.Parallel(64, 64):
                v9[i, j] = T.cast(T.cast(v8[i, j], "float32"), "float32")
            fn_1(v9, A, B, scratch_0, by, bx, 0, v10)
            for i, j in T.Parallel(64, 64):
                if by*64+i < 1 and bx*64+j < 3709:
                    C[by*64+i, bx*64+j] = v10[i, j]
    return impl


for name in ['typed_kernel_0']:
 for option,config in [('default',{}),('no_vector',{'tirx.disable_vectorize':True}),('no_async',{'tl.enable_async_copy':False})]:
  try:
   ir=globals()[name].get_tir();tilelang.compile(ir,target='cuda',pass_configs=config)
   print(json.dumps({'kernel':name,'option':option,'result':'PASS'}),flush=True)
  except Exception as e:
   print(json.dumps({'kernel':name,'option':option,'result':'FAIL','error':str(e)[-1800:]}),flush=True)
