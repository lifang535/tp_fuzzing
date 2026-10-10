"""Element types of slice programs and the exact value domains they carry.

Slice inputs are dyadic rationals k * 2**-frac with small |k|. Every slice
step transfers a static domain (lo, hi, frac), and a step is only emitted when
its result is exactly representable in the dtype it is computed or stored in
(YARPGen-style range tracking). The float64/int64 reference is then exact, so
outputs are compared bit-for-bit: a mismatch is never rounding noise.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DType:
    name: str
    kind: str          # 'float', 'int' or 'uint'
    bits: int
    mantissa: int      # significant bits of a float, implicit bit included
    maxval: float      # largest finite value
    triton: str
    tilelang: str
    torch: str         # torch storage dtype; '' when it cannot be a buffer

    @property
    def minval(self):
        if self.kind == 'float':
            return -self.maxval
        if self.kind == 'uint':
            return 0
        return -self.maxval - 1

    @property
    def is_float(self):
        return self.kind == 'float'

    @property
    def is_fp8(self):
        return self.name.startswith('f8')


DTYPES = {d.name: d for d in (
    DType('bf16', 'float', 16, 8, 3.38e38, 'tl.bfloat16', 'bfloat16', 'bfloat16'),
    DType('f16', 'float', 16, 11, 65504.0, 'tl.float16', 'float16', 'float16'),
    DType('f32', 'float', 32, 24, 3.4e38, 'tl.float32', 'float32', 'float32'),
    DType('f64', 'float', 64, 53, 1.7e308, 'tl.float64', 'float64', 'float64'),
    DType('f8e4', 'float', 8, 4, 448.0, 'tl.float8e4nv', 'float8_e4m3', 'float8_e4m3fn'),
    DType('f8e5', 'float', 8, 3, 57344.0, 'tl.float8e5', 'float8_e5m2', 'float8_e5m2'),
    DType('i8', 'int', 8, 0, 127, 'tl.int8', 'int8', 'int8'),
    DType('i16', 'int', 16, 0, 32767, 'tl.int16', 'int16', 'int16'),
    DType('i32', 'int', 32, 0, 2 ** 31 - 1, 'tl.int32', 'int32', 'int32'),
    DType('i64', 'int', 64, 0, 2 ** 63 - 1, 'tl.int64', 'int64', 'int64'),
    DType('u8', 'uint', 8, 0, 255, 'tl.uint8', 'uint8', 'uint8'),
    DType('u16', 'uint', 16, 0, 65535, 'tl.uint16', 'uint16', ''),
    DType('u32', 'uint', 32, 0, 2 ** 32 - 1, 'tl.uint32', 'uint32', ''),
    DType('u64', 'uint', 64, 0, 2 ** 64 - 1, 'tl.uint64', 'uint64', ''),
)}

FLOATS = tuple(n for n, d in DTYPES.items() if d.is_float)
INTEGERS = tuple(n for n, d in DTYPES.items() if not d.is_float)
STORAGE = tuple(n for n, d in DTYPES.items() if d.torch)
# Arithmetic dtypes: fp8 values are converted, loaded, stored and fed to MMA,
# but neither DSL computes elementwise in fp8.
ARITHMETIC = tuple(n for n in DTYPES if not DTYPES[n].is_fp8)


@dataclass(frozen=True)
class Domain:
    """Values k * 2**-frac with lo <= value <= hi."""
    lo: float
    hi: float
    frac: int = 0

    @property
    def magnitude(self):
        return max(abs(self.lo), abs(self.hi))

    def fits(self, dtype):
        d = DTYPES[dtype]
        if self.lo > self.hi:
            return False
        if not d.is_float:
            return self.frac == 0 and d.minval <= self.lo and self.hi <= d.maxval
        # A numerator below 2**mantissa needs no rounding at any exponent the
        # small domains reach; the magnitude bound keeps fp8 finite.
        return self.magnitude * 2 ** self.frac <= 2 ** d.mantissa and self.magnitude <= d.maxval

    def join(self, other):
        return Domain(min(self.lo, other.lo), max(self.hi, other.hi), max(self.frac, other.frac))


def bit_bound(domain):
    """A domain containing every bitwise and/or/xor of two values in domain."""
    width = max(1, math.ceil(math.log2(domain.magnitude + 1)))
    if domain.lo >= 0:
        return Domain(0, 2 ** width - 1)
    return Domain(-2 ** width, 2 ** width - 1)


def value_regimes():
    """Named input domains a slice draws its inputs from."""
    return {
        'tiny': Domain(-2, 2),
        'small': Domain(-8, 8),
        'nonneg': Domain(0, 15),
        'half': Domain(-4, 4, 1),
        'quarter': Domain(-2, 2, 2),
        'unit': Domain(-1, 1),
    }
