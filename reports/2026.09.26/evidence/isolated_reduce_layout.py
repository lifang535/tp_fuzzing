import tilelang
import tilelang.language as T
import json,traceback
dtype="float16"


@tilelang.jit
def typed_kernel_0():
    @T.macro
    def fn_0(arg0, A, B, by, bx, iv, fn_out):
        v1_stat = T.alloc_fragment((64,), "float32")
        v1_wide = T.alloc_fragment((64, 256), "float32")
        v5_stat = T.alloc_fragment((256,), "float32")
        v5_wide = T.alloc_fragment((64, 256), "float32")
        v7_stat = T.alloc_fragment((1,), "float32")
        v7_wide = T.alloc_fragment((1, 256), "float32")
        v1 = T.alloc_fragment((64, 1), "float32")
        v2 = T.alloc_fragment((64, 1), "float32")
        v12 = T.alloc_fragment((64, 256), "float32")
        v3 = T.alloc_fragment((64, 256), "float32")
        v10 = T.alloc_fragment((64, 256), "float32")
        v4 = T.alloc_fragment((64, 256), "float32")
        v5 = T.alloc_fragment((1, 256), "float32")
        v6 = T.alloc_fragment((1, 256), "float32")
        v7 = T.alloc_fragment((1, 256), "float32")
        v8 = T.alloc_fragment((1, 256), "float32")
        v9 = T.alloc_fragment((64, 256), "float32")
        v11 = T.alloc_fragment((64, 256), "float32")
        v13 = T.alloc_fragment((64, 256), "float32")
        for i, j in T.Parallel(64, 256):
            v1_wide[i, j] = T.cast(arg0[i, j], "float32")
        T.reduce_min(v1_wide, v1_stat, dim=1, clear=True)
        for i, j in T.Parallel(64, 1):
            v1[i, j] = T.cast(v1_stat[i], "float32")
        for i, j in T.Parallel(64, 1):
            v2[i, j] = T.cast(T.cast(T.cast(T.cast(v1[i, 0], "float32"), "float16"), "float32"), "float32")
        if bx % 4 == 2:
            T.copy(arg0, v3)
            for i, j in T.Parallel(64, 256):
                v4[i, j] = T.cast(T.cos(T.cast(v3[i, j], "float32")), "float32")
            for i, j in T.Parallel(64, 256):
                v5_wide[i, j] = T.cast(v4[i, j], "float32")
            T.reduce_min(v5_wide, v5_stat, dim=0, clear=True)
            for i, j in T.Parallel(1, 256):
                v5[i, j] = T.cast(v5_stat[j], "float32")
            for i, j in T.Parallel(1, 256):
                v6[i, j] = T.cast(T.cos(T.cast(v5[0, j], "float32")), "float32")
            for i, j in T.Parallel(1, 256):
                v7_wide[i, j] = T.cast(v6[i, j], "float32")
            T.reduce_min(v7_wide, v7_stat, dim=1, clear=True)
            for i, j in T.Parallel(1, 256):
                v7[i, j] = T.cast(v7_stat[i], "float32")
            for i, j in T.Parallel(1, 256):
                v8[i, j] = T.cast(T.floor(T.cast(v7[0, j], "float32")), "float32")
            for i, j in T.Parallel(64, 256):
                v9[i, j] = T.cast(T.cast(v5[0, j], "float32"), "float32")
            T.copy(v9, v12)
        else:
            T.copy(arg0, v10)
            for i, j in T.Parallel(64, 256):
                v11[i, j] = T.cast(T.cast(v10[i, j], "float32"), "float32")
            T.copy(v11, v12)
        for i, j in T.Parallel(64, 256):
            v13[i, j] = T.cast(T.cast(v12[i, j], "float32"), "float32")
        T.copy(v13, fn_out)
    @T.prim_func
    def impl(A: T.Buffer((47664556,), dtype), B: T.Buffer((18583652,), dtype), C: T.Buffer((7315, 2849), dtype)):
        with T.Kernel(115, 12, threads=256) as (by, bx):
            v3_stat = T.alloc_fragment((256,), "float32")
            v3_wide = T.alloc_fragment((64, 256), "float32")
            v6_stat = T.alloc_fragment((1,), "float32")
            v6_wide = T.alloc_fragment((1, 256), "float32")
            v1 = T.alloc_fragment((64, 256), "float32")
            v2 = T.alloc_fragment((64, 256), "float32")
            v3 = T.alloc_fragment((1, 256), "float32")
            v4 = T.alloc_fragment((1, 256), "float32")
            v5 = T.alloc_fragment((1, 256), "float32")
            v6 = T.alloc_fragment((1, 256), "float32")
            v7 = T.alloc_fragment((1, 256), "float32")
            v12 = T.alloc_fragment((64, 256), "float32")
            v8 = T.alloc_fragment((64, 256), "float32")
            v9 = T.alloc_fragment((64, 256), "float32")
            v10 = T.alloc_fragment((64, 256), "float32")
            v11 = T.alloc_fragment((64, 256), "float32")
            v13 = T.alloc_fragment((64, 256), "float32")
            v14 = T.alloc_fragment((64, 256), "float32")
            As = T.alloc_shared((64, 8), dtype)
            Bs = T.alloc_shared((8, 256), dtype)
            T.clear(v1)
            for ki in T.serial(815):
                for i, j in T.Parallel(64, 8):
                    As[i, j] = T.if_then_else(by*64+i < 7315 and ki*8+j < 6516, A[0+(by*64+i)*6516+(ki*8+j)*1], T.cast(0, dtype))
                for i, j in T.Parallel(8, 256):
                    Bs[i, j] = T.if_then_else(ki*8+i < 6516 and bx*256+j < 2849, B[7+(ki*8+i)*2852+(bx*256+j)*1], T.cast(0, dtype))
                T.gemm(As, Bs, v1)
            for i, j in T.Parallel(64, 256):
                v2[i, j] = T.cast(T.ceil(T.cast(v1[i, j], "float32")), "float32")
            for i, j in T.Parallel(64, 256):
                v3_wide[i, j] = T.cast(v2[i, j], "float32")
            T.reduce_min(v3_wide, v3_stat, dim=0, clear=True)
            for i, j in T.Parallel(1, 256):
                v3[i, j] = T.cast(v3_stat[j], "float32")
            for i, j in T.Parallel(1, 256):
                v4[i, j] = T.cast(T.cast(v3[0, j], "float32") * 9.173141204464626, "float32")
            for i, j in T.Parallel(1, 256):
                v5[i, j] = T.cast(T.log(T.max(T.abs(T.cast(v4[0, j], "float32")), T.float32(0.001))), "float32")
            for i, j in T.Parallel(1, 256):
                v6_wide[i, j] = T.cast(v5[i, j], "float32")
            T.reduce_sum(v6_wide, v6_stat, dim=1, clear=True)
            for i, j in T.Parallel(1, 256):
                v6[i, j] = T.cast(v6_stat[i], "float32")
            for i, j in T.Parallel(1, 256):
                v7[i, j] = T.cast(T.cast(T.cast(T.cast(v4[0, j], "float32"), "float32"), "float32"), "float32")
            T.copy(v2, v12)
            for iter_v12 in T.serial(2):
                T.copy(v12, v8)
                for i, j in T.Parallel(64, 256):
                    v9[i, j] = T.cast(T.cast(v8[i, j], "float32"), "float32")
                for i, j in T.Parallel(64, 256):
                    v10[i, j] = T.cast(T.cast(v8[i, j], "float32"), "float32")
                for i, j in T.Parallel(64, 256):
                    v11[i, j] = T.cast(T.cast(v10[i, j], "float32") + T.cast(v8[i, j], "float32"), "float32")
                T.copy(v11, v12)
            for i, j in T.Parallel(64, 256):
                v13[i, j] = T.cast(T.cast(v2[i, j], "float32"), "float32")
            fn_0(v13, A, B, by, bx, 0, v14)
            for i, j in T.Parallel(64, 256):
                if by*64+i < 7315 and bx*256+j < 2849:
                    C[by*64+i, bx*256+j] = v14[i, j]
    return impl


for name in ['typed_kernel_0']:
 for option,config in [('default',{}),('no_vector',{'tirx.disable_vectorize':True}),('no_async',{'tl.enable_async_copy':False})]:
  try:
   ir=globals()[name].get_tir();tilelang.compile(ir,target='cuda',pass_configs=config)
   print(json.dumps({'kernel':name,'option':option,'result':'PASS'}),flush=True)
  except Exception as e:
   print(json.dumps({'kernel':name,'option':option,'result':'FAIL','error':str(e)[-1800:]}),flush=True)
