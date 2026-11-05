import triton
import triton.language as tl


@triton.jit
def e2m1_triton_ordinal(v, n):
    sgn = (tl.cast(v, tl.uint32, bitcast=True) >> 31) != 0
    av = tl.minimum(tl.abs(v), 6.0)
    mag = (
        (av > 0.25).to(tl.int32)
        + (av > 0.75).to(tl.int32)
        + (av > 1.25).to(tl.int32)
        + (av > 1.75).to(tl.int32)
        + (av > 2.5).to(tl.int32)
        + (av > 3.5).to(tl.int32)
        + (av > 5.0).to(tl.int32)
    )
    return tl.where(sgn, 7 - mag, 8 + mag)


@triton.jit
def ue4m3_triton_ordinal(v, n):
    u8 = tl.cast(
        tl.minimum(tl.maximum(v, 2.0**-9), 448.0).to(tl.float32), tl.float8e4nv
    )
    key = tl.cast(u8, tl.uint8, bitcast=True).to(tl.int32) - 1
    return tl.minimum(tl.maximum(key, 0), n - 1)


@triton.jit
def e8m0_triton_ordinal(v, n):
    v = tl.minimum(tl.maximum(v, 2.0**-127), 2.0**127).to(tl.float32)
    bits = tl.cast(v, tl.uint32, bitcast=True)
    exp8 = (bits >> 23).to(tl.int32)
    mant = (bits & 0x7FFFFF).to(tl.int32)
    return tl.where(
        exp8 == 0,
        (mant > 4194304).to(tl.int32),
        tl.minimum(exp8 + (mant >= 4194304).to(tl.int32), n - 1),
    )


@triton.jit
def e4m3_triton_ordinal(v, n):
    u8 = tl.cast(tl.minimum(tl.maximum(v, -448.0), 448.0).to(tl.float32), tl.float8e4nv)
    b = tl.cast(u8, tl.uint8, bitcast=True).to(tl.int32)
    return tl.minimum(tl.maximum(tl.where(b < 128, b + 127, 254 - b), 0), n - 1)
