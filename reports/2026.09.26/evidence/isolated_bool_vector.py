import tilelang
import tilelang.language as T
import json,traceback
dtype="float16"


@tilelang.jit
def extended_0_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 544), "int32"), mem3: T.Buffer((3, 288), "float16"), mem4: T.Buffer((3, 36), "float16"), out_e64: T.Buffer((3, 288), "float32"), out_e18: T.Buffer((3, 544), "int32"), out_e24: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=32) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((32, 16), "bool")
            e7 = T.alloc_fragment((32, 16), "float32")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "float16")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "float16")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "bool")
            e16 = T.alloc_fragment((32, 16), "int32")
            e17 = T.alloc_fragment((32, 16), "int32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "bool")
            e21 = T.alloc_fragment((32, 16), "int32")
            e22 = T.alloc_fragment((1,), "int32")
            e23 = T.alloc_fragment((32, 16), "bool")
            e24 = T.alloc_fragment((32, 16), "bool")
            e25 = T.alloc_fragment((32, 16), "float16")
            e26 = T.alloc_fragment((32, 16), "float16")
            e27 = T.alloc_fragment((32, 16), "float16")
            e28 = T.alloc_fragment((16, 32), "float16")
            e29 = T.alloc_fragment((16, 16), "float32")
            e30 = T.alloc_fragment((16, 16), "float32")
            e31 = T.alloc_fragment((16,), "float32")
            e32 = T.alloc_fragment((16, 1), "float32")
            e33 = T.alloc_fragment((16, 16), "float32")
            e34 = T.alloc_fragment((16, 16), "float32")
            e35 = T.alloc_fragment((16, 16), "float32")
            e36 = T.alloc_fragment((16, 16), "int32")
            e37 = T.alloc_fragment((16, 16), "bool")
            e38 = T.alloc_fragment((16, 16), "float16")
            e39 = T.alloc_fragment((16, 16), "float16")
            e40 = T.alloc_fragment((16, 16), "float32")
            e41 = T.alloc_fragment((16, 16), "float32")
            e42 = T.alloc_fragment((16, 16), "float32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((1,), "float16")
            e46 = T.alloc_fragment((1,), "int32")
            e47 = T.alloc_fragment((1,), "bool")
            e48 = T.alloc_fragment((1,), "float16")
            e49 = T.alloc_fragment((1,), "int32")
            e50 = T.alloc_fragment((1,), "bool")
            e51 = T.alloc_fragment((1,), "float16")
            e52 = T.alloc_fragment((1,), "float16")
            e53 = T.alloc_fragment((1,), "float16")
            e54 = T.alloc_fragment((1,), "float16")
            e55 = T.alloc_fragment((1,), "float16")
            e56 = T.alloc_fragment((1,), "float16")
            e57 = T.alloc_fragment((1,), "float16")
            e58 = T.alloc_fragment((1,), "float16")
            e59 = T.alloc_fragment((1,), "float16")
            e60 = T.alloc_fragment((16, 16), "float32")
            e61 = T.alloc_fragment((16, 16), "float16")
            e62 = T.alloc_fragment((16, 16), "float16")
            e63 = T.alloc_fragment((16, 16), "float32")
            e64 = T.alloc_fragment((16, 16), "float32")
            e30_a_shared = T.alloc_shared((16, 32), "float16")
            e30_b_shared = T.alloc_shared((32, 16), "float16")
            e31_wide = T.alloc_fragment((16, 16), "float32")
            e31_nan = T.alloc_fragment((16, 16), "int32")
            e31_nan_count = T.alloc_fragment((16,), "int32")
            e40_a_shared = T.alloc_shared((16, 16), "float16")
            e40_b_shared = T.alloc_shared((16, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 44) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((i * 16 + j) + 370) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(T.if_then_else((e6[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 16 + e5[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast((e4[i, j] * e7[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(e8[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e9[i, j] + e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(e10[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(0.5, "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast((e11[i, j] * e12[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((((511 - (i * 16 + j)) + 389) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(T.if_then_else((e15[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem2[bid, 16 + e14[i, j] * 1], T.cast(0, "int32")), "int32")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast((e16[i, j] ^ e17[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast((e18[i, j] < e19[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((((511 - (i * 16 + j)) + 166) % 512), "int32")
            e22[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast((e21[i, j] < e22[0]), "bool")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((e20[i, j] and e23[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(-0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast(T.if_then_else(e24[i, j], e13[i, j], e25[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(e26[i, j], "float16")
            for i, j in T.Parallel(16, 32):
                e28[i, j] = T.cast(e27[j, i], "float16")
            for i, j in T.Parallel(16, 16):
                e29[i, j] = T.cast(0.0, "float32")
            T.copy(e28, e30_a_shared)
            T.copy(e10, e30_b_shared)
            T.copy(e29, e30)
            T.gemm(e30_a_shared, e30_b_shared, e30)
            for i, j in T.Parallel(16, 16):
                e31_wide[i, j] = T.cast(e30[i, j], 'float32')
            T.reduce_max(e31_wide, e31, dim=1, clear=True)
            for i, j in T.Parallel(16, 16):
                e31_nan[i, j] = T.if_then_else(T.isnan(e31_wide[i, j]), 1, 0)
            T.reduce_sum(e31_nan, e31_nan_count, dim=1, clear=True)
            for i in T.Parallel(16):
                e31[i] = T.if_then_else(e31_nan_count[i] > 0, T.cast(float('nan'), 'float32'), e31[i])
            for i, j in T.Parallel(16, 1):
                e32[i, 0] = T.cast(e31[(i * 1 + j)], "float32")
            for i, j in T.Parallel(16, 16):
                e33[i, j] = T.cast((e30[i, j] - e32[i, 0]), "float32")
            for i, j in T.Parallel(16, 16):
                e34[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(16, 16):
                e35[i, j] = T.cast((e33[i, j] * e34[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                e36[i, j] = T.cast((((i * 16 + j) + 125) % 256), "int32")
            for i, j in T.Parallel(16, 16):
                e37[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 16):
                e38[i, j] = T.cast(T.if_then_else((e37[i, j] and e36[i, j] >= 0 and e36[i, j] < 256), mem3[bid, 16 + e36[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 16):
                e39[i, j] = T.cast(e35[i, j], "float16")
            T.copy(e39, e40_a_shared)
            T.copy(e38, e40_b_shared)
            T.copy(e29, e40)
            T.gemm(e40_a_shared, e40_b_shared, e40)
            for i, j in T.Parallel(16, 16):
                e41[i, j] = T.cast(e40[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            for i, j in T.Parallel(16, 16):
                e42[i, j] = T.cast(e41[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            e43[0] = T.cast(0, "int32")
            e44[0] = T.cast(True, "bool")
            e45[0] = T.cast(T.if_then_else((e44[0] and e43[0] >= 0 and e43[0] < 4), mem4[bid, 16 + e43[0] * 1], T.cast(0, "float16")), "float16")
            e46[0] = T.cast(1, "int32")
            e47[0] = T.cast(True, "bool")
            e48[0] = T.cast(T.if_then_else((e47[0] and e46[0] >= 0 and e46[0] < 4), mem4[bid, 16 + e46[0] * 1], T.cast(0, "float16")), "float16")
            e49[0] = T.cast(2, "int32")
            e50[0] = T.cast(True, "bool")
            e51[0] = T.cast(T.if_then_else((e50[0] and e49[0] >= 0 and e49[0] < 4), mem4[bid, 16 + e49[0] * 1], T.cast(0, "float16")), "float16")
            e52[0] = T.cast(1.0, "float16")
            e53[0] = T.cast((e45[0] * e52[0]), "float16")
            e54[0] = T.cast(0.125, "float16")
            e55[0] = T.cast((e48[0] * e54[0]), "float16")
            e56[0] = T.cast(-0.125, "float16")
            e57[0] = T.cast((e51[0] * e56[0]), "float16")
            e58[0] = T.cast((e53[0] * e55[0] + e57[0]), "float16")
            e59[0] = T.cast((e58[0] * e55[0] + e57[0]), "float16")
            for i, j in T.Parallel(16, 16):
                e60[i, j] = T.cast((e42[i, j] + e40[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                e61[i, j] = T.cast(e60[i, j], "float16")
            for i, j in T.Parallel(16, 16):
                e62[i, j] = T.cast((e61[i, j] - e39[i, j]), "float16")
            for i, j in T.Parallel(16, 16):
                e63[i, j] = T.cast(e62[i, j], "float32")
            for i, j in T.Parallel(16, 16):
                e64[i, j] = T.cast((e63[i, j] * e63[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                out_e64[bid, 16 + (i * 16 + j)] = e64[i, j]
            for i, j in T.Parallel(32, 16):
                out_e18[bid, 16 + (i * 16 + j)] = e18[i, j]
            for i, j in T.Parallel(32, 16):
                out_e24[bid, 16 + (i * 16 + j)] = e24[i, j]
    return impl

@tilelang.jit
def extended_0_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 544), "int32"), mem3: T.Buffer((3, 288), "float16"), mem4: T.Buffer((3, 36), "float16"), out_e64: T.Buffer((3, 288), "float32"), out_e18: T.Buffer((3, 544), "int32"), out_e24: T.Buffer((3, 544), "bool"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float16"), out_e30: T.Buffer((3, 288), "float32"), out_e35: T.Buffer((3, 288), "float32"), out_e58: T.Buffer((3, 33), "float16"), out_e59: T.Buffer((3, 33), "float16"), out_e22: T.Buffer((3, 33), "int32"), out_e47: T.Buffer((3, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=32) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((32, 16), "bool")
            e7 = T.alloc_fragment((32, 16), "float32")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "float16")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "float16")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "bool")
            e16 = T.alloc_fragment((32, 16), "int32")
            e17 = T.alloc_fragment((32, 16), "int32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "bool")
            e21 = T.alloc_fragment((32, 16), "int32")
            e22 = T.alloc_fragment((1,), "int32")
            e23 = T.alloc_fragment((32, 16), "bool")
            e24 = T.alloc_fragment((32, 16), "bool")
            e25 = T.alloc_fragment((32, 16), "float16")
            e26 = T.alloc_fragment((32, 16), "float16")
            e27 = T.alloc_fragment((32, 16), "float16")
            e28 = T.alloc_fragment((16, 32), "float16")
            e29 = T.alloc_fragment((16, 16), "float32")
            e30 = T.alloc_fragment((16, 16), "float32")
            e31 = T.alloc_fragment((16,), "float32")
            e32 = T.alloc_fragment((16, 1), "float32")
            e33 = T.alloc_fragment((16, 16), "float32")
            e34 = T.alloc_fragment((16, 16), "float32")
            e35 = T.alloc_fragment((16, 16), "float32")
            e36 = T.alloc_fragment((16, 16), "int32")
            e37 = T.alloc_fragment((16, 16), "bool")
            e38 = T.alloc_fragment((16, 16), "float16")
            e39 = T.alloc_fragment((16, 16), "float16")
            e40 = T.alloc_fragment((16, 16), "float32")
            e41 = T.alloc_fragment((16, 16), "float32")
            e42 = T.alloc_fragment((16, 16), "float32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((1,), "float16")
            e46 = T.alloc_fragment((1,), "int32")
            e47 = T.alloc_fragment((1,), "bool")
            e48 = T.alloc_fragment((1,), "float16")
            e49 = T.alloc_fragment((1,), "int32")
            e50 = T.alloc_fragment((1,), "bool")
            e51 = T.alloc_fragment((1,), "float16")
            e52 = T.alloc_fragment((1,), "float16")
            e53 = T.alloc_fragment((1,), "float16")
            e54 = T.alloc_fragment((1,), "float16")
            e55 = T.alloc_fragment((1,), "float16")
            e56 = T.alloc_fragment((1,), "float16")
            e57 = T.alloc_fragment((1,), "float16")
            e58 = T.alloc_fragment((1,), "float16")
            e59 = T.alloc_fragment((1,), "float16")
            e60 = T.alloc_fragment((16, 16), "float32")
            e61 = T.alloc_fragment((16, 16), "float16")
            e62 = T.alloc_fragment((16, 16), "float16")
            e63 = T.alloc_fragment((16, 16), "float32")
            e64 = T.alloc_fragment((16, 16), "float32")
            e30_a_shared = T.alloc_shared((16, 32), "float16")
            e30_b_shared = T.alloc_shared((32, 16), "float16")
            e31_wide = T.alloc_fragment((16, 16), "float32")
            e31_nan = T.alloc_fragment((16, 16), "int32")
            e31_nan_count = T.alloc_fragment((16,), "int32")
            e40_a_shared = T.alloc_shared((16, 16), "float16")
            e40_b_shared = T.alloc_shared((16, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 44) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((i * 16 + j) + 370) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(T.if_then_else((e6[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 16 + e5[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast((e4[i, j] * e7[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(e8[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e9[i, j] + e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(e10[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(0.5, "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast((e11[i, j] * e12[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((((511 - (i * 16 + j)) + 389) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(T.if_then_else((e15[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem2[bid, 16 + e14[i, j] * 1], T.cast(0, "int32")), "int32")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast((e16[i, j] ^ e17[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast((e18[i, j] < e19[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((((511 - (i * 16 + j)) + 166) % 512), "int32")
            e22[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast((e21[i, j] < e22[0]), "bool")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((e20[i, j] and e23[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(-0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast(T.if_then_else(e24[i, j], e13[i, j], e25[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(e26[i, j], "float16")
            for i, j in T.Parallel(16, 32):
                e28[i, j] = T.cast(e27[j, i], "float16")
            for i, j in T.Parallel(16, 16):
                e29[i, j] = T.cast(0.0, "float32")
            T.copy(e28, e30_a_shared)
            T.copy(e10, e30_b_shared)
            T.copy(e29, e30)
            T.gemm(e30_a_shared, e30_b_shared, e30)
            for i, j in T.Parallel(16, 16):
                e31_wide[i, j] = T.cast(e30[i, j], 'float32')
            T.reduce_max(e31_wide, e31, dim=1, clear=True)
            for i, j in T.Parallel(16, 16):
                e31_nan[i, j] = T.if_then_else(T.isnan(e31_wide[i, j]), 1, 0)
            T.reduce_sum(e31_nan, e31_nan_count, dim=1, clear=True)
            for i in T.Parallel(16):
                e31[i] = T.if_then_else(e31_nan_count[i] > 0, T.cast(float('nan'), 'float32'), e31[i])
            for i, j in T.Parallel(16, 1):
                e32[i, 0] = T.cast(e31[(i * 1 + j)], "float32")
            for i, j in T.Parallel(16, 16):
                e33[i, j] = T.cast((e30[i, j] - e32[i, 0]), "float32")
            for i, j in T.Parallel(16, 16):
                e34[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(16, 16):
                e35[i, j] = T.cast((e33[i, j] * e34[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                e36[i, j] = T.cast((((i * 16 + j) + 125) % 256), "int32")
            for i, j in T.Parallel(16, 16):
                e37[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 16):
                e38[i, j] = T.cast(T.if_then_else((e37[i, j] and e36[i, j] >= 0 and e36[i, j] < 256), mem3[bid, 16 + e36[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 16):
                e39[i, j] = T.cast(e35[i, j], "float16")
            T.copy(e39, e40_a_shared)
            T.copy(e38, e40_b_shared)
            T.copy(e29, e40)
            T.gemm(e40_a_shared, e40_b_shared, e40)
            for i, j in T.Parallel(16, 16):
                e41[i, j] = T.cast(e40[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            for i, j in T.Parallel(16, 16):
                e42[i, j] = T.cast(e41[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            e43[0] = T.cast(0, "int32")
            e44[0] = T.cast(True, "bool")
            e45[0] = T.cast(T.if_then_else((e44[0] and e43[0] >= 0 and e43[0] < 4), mem4[bid, 16 + e43[0] * 1], T.cast(0, "float16")), "float16")
            e46[0] = T.cast(1, "int32")
            e47[0] = T.cast(True, "bool")
            e48[0] = T.cast(T.if_then_else((e47[0] and e46[0] >= 0 and e46[0] < 4), mem4[bid, 16 + e46[0] * 1], T.cast(0, "float16")), "float16")
            e49[0] = T.cast(2, "int32")
            e50[0] = T.cast(True, "bool")
            e51[0] = T.cast(T.if_then_else((e50[0] and e49[0] >= 0 and e49[0] < 4), mem4[bid, 16 + e49[0] * 1], T.cast(0, "float16")), "float16")
            e52[0] = T.cast(1.0, "float16")
            e53[0] = T.cast((e45[0] * e52[0]), "float16")
            e54[0] = T.cast(0.125, "float16")
            e55[0] = T.cast((e48[0] * e54[0]), "float16")
            e56[0] = T.cast(-0.125, "float16")
            e57[0] = T.cast((e51[0] * e56[0]), "float16")
            e58[0] = T.cast((e53[0] * e55[0] + e57[0]), "float16")
            e59[0] = T.cast((e58[0] * e55[0] + e57[0]), "float16")
            for i, j in T.Parallel(16, 16):
                e60[i, j] = T.cast((e42[i, j] + e40[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                e61[i, j] = T.cast(e60[i, j], "float16")
            for i, j in T.Parallel(16, 16):
                e62[i, j] = T.cast((e61[i, j] - e39[i, j]), "float16")
            for i, j in T.Parallel(16, 16):
                e63[i, j] = T.cast(e62[i, j], "float32")
            for i, j in T.Parallel(16, 16):
                e64[i, j] = T.cast((e63[i, j] * e63[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                out_e64[bid, 16 + (i * 16 + j)] = e64[i, j]
            for i, j in T.Parallel(32, 16):
                out_e18[bid, 16 + (i * 16 + j)] = e18[i, j]
            for i, j in T.Parallel(32, 16):
                out_e24[bid, 16 + (i * 16 + j)] = e24[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(16, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(16, 16):
                out_e35[bid, 16 + (i * 16 + j)] = e35[i, j]
            out_e58[bid, 16 + 0] = e58[0]
            out_e59[bid, 16 + 0] = e59[0]
            out_e22[bid, 16 + 0] = e22[0]
            out_e47[bid, 16 + 0] = e47[0]
    return impl


for name in ['extended_0_0', 'extended_0_1']:
 for option,config in [('default',{}),('no_vector',{'tirx.disable_vectorize':True}),('no_async',{'tl.enable_async_copy':False})]:
  try:
   ir=globals()[name].get_tir();tilelang.compile(ir,target='cuda',pass_configs=config)
   print(json.dumps({'kernel':name,'option':option,'result':'PASS'}),flush=True)
  except Exception as e:
   print(json.dumps({'kernel':name,'option':option,'result':'FAIL','error':str(e)[-1800:]}),flush=True)
