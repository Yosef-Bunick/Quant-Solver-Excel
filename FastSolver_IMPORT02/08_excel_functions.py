"""
excel_functions.py — Python equivalents of the 15 Excel functions used in
the actuarial workbook. Each function has two flavors:

  exact_*  : matches Excel behavior bit-for-bit (for validation)
  smooth_* : differentiable approximation parameterized by sharpness k
             (for AutoDiff / homotopy continuation)

Functions covered:
  ABS, AND, AVERAGE, IF, IFERROR, ISNUMBER, LOOKUP, NA,
  NOT, OFFSET, OR, PRODUCT, STDEV, SUM, SUMPRODUCT

Plus the special idiom LOOKUP(2, 1/ISNUMBER(range), range) — "last non-blank".
"""

from __future__ import annotations
import math
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------------
# Sentinel for Excel's NA() / #N/A
# ---------------------------------------------------------------------------
class ExcelNA:
    """Singleton representing Excel's #N/A error."""
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst
    def __repr__(self): return "#N/A"
    def __bool__(self): return False

NA = ExcelNA()

def is_na(x):
    return isinstance(x, ExcelNA) or (isinstance(x, float) and math.isnan(x))

# ---------------------------------------------------------------------------
# EXACT IMPLEMENTATIONS
# Match Excel behavior precisely. Used for validation against live Excel.
# ---------------------------------------------------------------------------

def exact_abs(x):
    if is_na(x): return NA
    return abs(x)

def exact_and(*args):
    for a in args:
        if is_na(a): return NA
        if not bool(a): return False
    return True

def exact_or(*args):
    for a in args:
        if is_na(a): return NA
        if bool(a): return True
    return False

def exact_not(x):
    if is_na(x): return NA
    return not bool(x)

def exact_if(cond, t, f):
    if is_na(cond): return NA
    return t if bool(cond) else f

def exact_iferror(value, fallback):
    if is_na(value): return fallback
    return value

def exact_isnumber(x):
    if is_na(x): return False
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def exact_na():
    return NA

def _filter_nums(seq):
    return [v for v in seq if exact_isnumber(v)]

def exact_sum(*args):
    flat = []
    for a in args:
        if isinstance(a, (list, tuple)):
            flat.extend(_flatten(a))
        else:
            flat.append(a)
    nums = _filter_nums(flat)
    return sum(nums) if nums else 0

def _flatten(seq):
    out = []
    for v in seq:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten(v))
        else:
            out.append(v)
    return out

def exact_average(*args):
    nums = _filter_nums(_flatten(args))
    if not nums: return NA
    return sum(nums) / len(nums)

def exact_product(*args):
    nums = _filter_nums(_flatten(args))
    if not nums: return 0
    p = 1.0
    for n in nums: p *= n
    return p

def exact_stdev(*args):
    """Excel STDEV — sample stddev with n-1 in denominator."""
    nums = _filter_nums(_flatten(args))
    if len(nums) < 2: return NA
    m = sum(nums) / len(nums)
    var = sum((x - m) ** 2 for x in nums) / (len(nums) - 1)
    return math.sqrt(var)

def exact_sumproduct(*arrays):
    """SUMPRODUCT — elementwise multiply then sum. Non-numerics treated as 0."""
    arrays = [_flatten(a) if isinstance(a, (list, tuple)) else [a] for a in arrays]
    n = len(arrays[0])
    total = 0.0
    for i in range(n):
        prod = 1.0
        valid = True
        for arr in arrays:
            v = arr[i]
            if exact_isnumber(v):
                prod *= v
            elif isinstance(v, bool):
                prod *= (1.0 if v else 0.0)
            else:
                # non-numeric → treat as 0 per Excel
                prod = 0.0
                break
        total += prod
    return total

def exact_lookup(lookup_value, lookup_vector, result_vector=None):
    """
    Excel LOOKUP: searches lookup_vector for the largest value <= lookup_value,
    returns corresponding result_vector value (or lookup_vector if result omitted).

    Special idiom: LOOKUP(2, 1/ISNUMBER(range), range) returns last non-blank.
    Because 1/TRUE = 1 and 1/FALSE = #DIV/0!, and LOOKUP ignores errors,
    it ends up finding the last "1" in the vector → last non-blank position.
    """
    lookup_vector = _flatten(lookup_vector)
    if result_vector is None:
        result_vector = lookup_vector
    else:
        result_vector = _flatten(result_vector)

    # Excel LOOKUP requires ascending sort; with division-by-zero errors it
    # walks the array and returns the LAST valid match <= lookup_value.
    best_idx = None
    for i, v in enumerate(lookup_vector):
        if exact_isnumber(v) and v <= lookup_value:
            best_idx = i
    if best_idx is None: return NA
    return result_vector[best_idx]

def exact_offset(reference, rows, cols, height=None, width=None):
    """
    OFFSET returns a reference shifted by (rows, cols).
    In pycel context, 'reference' arrives as a value or a sub-array.
    For our prototype we operate on already-resolved arrays passed by the graph.
    """
    # When pycel is the host, OFFSET resolution happens at graph build time.
    # This stub handles the value-level fallback.
    if isinstance(reference, (list, tuple)):
        flat = _flatten(reference)
        idx = int(rows) * (width or 1) + int(cols)
        if 0 <= idx < len(flat):
            return flat[idx]
        return NA
    return reference


# ---------------------------------------------------------------------------
# SMOOTH IMPLEMENTATIONS
# Differentiable. Each takes sharpness parameter k (higher = closer to exact).
# All operate on jax arrays / scalars.
# ---------------------------------------------------------------------------

def _sigmoid(x, k=10.0):
    """Smooth step. Approaches Heaviside as k → ∞."""
    return 1.0 / (1.0 + jnp.exp(-k * x))

def smooth_abs(x, eps=1e-6):
    """|x| ≈ √(x² + ε). Differentiable everywhere."""
    return jnp.sqrt(x * x + eps)

def smooth_max(a, b, eps=1e-6):
    """max(a,b) ≈ (a+b+√((a-b)²+ε))/2."""
    return 0.5 * (a + b + jnp.sqrt((a - b) ** 2 + eps))

def smooth_min(a, b, eps=1e-6):
    return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + eps))

def smooth_if(cond, t, f, k=10.0):
    """IF(cond > 0, t, f) ≈ σ(k·cond)·t + (1-σ)·f."""
    w = _sigmoid(cond, k)
    return w * t + (1.0 - w) * f

def smooth_and(*conds, k=10.0):
    """Smooth AND via product of sigmoids — all must be > 0."""
    result = jnp.array(1.0)
    for c in conds:
        result = result * _sigmoid(c, k)
    return result

def smooth_or(*conds, k=10.0):
    """De Morgan: OR = 1 - AND(NOT)."""
    result = jnp.array(1.0)
    for c in conds:
        result = result * (1.0 - _sigmoid(c, k))
    return 1.0 - result

def smooth_not(c, k=10.0):
    return 1.0 - _sigmoid(c, k)

def smooth_iferror(value, fallback, validity, k=10.0):
    """
    Continuous validity: validity > 0 means "no error".
    Caller must provide a smooth validity indicator (e.g. denom² + ε).
    """
    w = _sigmoid(validity, k)
    return w * value + (1.0 - w) * fallback

def smooth_isnumber(x, mask, k=10.0):
    """
    'Is this value numeric?' — in smooth land, this is encoded as a mask
    (1.0 if originally numeric, 0.0 if blank/text). At build time we pre-mask
    the array; this function just returns the mask for use in compositions.
    """
    return mask

def smooth_sum(values, mask=None):
    """Sum with optional smooth mask (mask close to 0 = excluded)."""
    if mask is None:
        return jnp.sum(values)
    return jnp.sum(values * mask)

def smooth_average(values, mask=None, eps=1e-6):
    if mask is None:
        return jnp.mean(values)
    total = jnp.sum(values * mask)
    count = jnp.sum(mask) + eps
    return total / count

def smooth_product(values, mask=None):
    if mask is None:
        return jnp.prod(values)
    # mask near 0 → factor → 1 (don't multiply by it)
    return jnp.prod(mask * values + (1.0 - mask))

def smooth_stdev(values, mask=None, eps=1e-6):
    """Sample stddev (n-1 in denom), smooth-mask aware."""
    if mask is None:
        m = jnp.mean(values)
        var = jnp.mean((values - m) ** 2) * len(values) / (len(values) - 1)
        return jnp.sqrt(var + eps)
    n_eff = jnp.sum(mask) + eps
    m = jnp.sum(values * mask) / n_eff
    sq = jnp.sum(((values - m) ** 2) * mask)
    var = sq / (n_eff - 1.0 + eps)
    return jnp.sqrt(var + eps)

def smooth_sumproduct(*arrays):
    """Elementwise product across arrays, then sum."""
    prod = arrays[0]
    for a in arrays[1:]:
        prod = prod * a
    return jnp.sum(prod)

def smooth_lookup_last_nonblank(values, mask, k=10.0):
    """
    The LOOKUP(2, 1/ISNUMBER(range), range) idiom = last non-blank.
    Smooth version: soft argmax favoring rightmost present index.

    Position score = mask[i] * exp(k * i)  → normalized to attention weights
    Returns Σ weight[i] * values[i].
    """
    n = values.shape[0]
    idx = jnp.arange(n, dtype=values.dtype)
    # log-space for stability
    log_score = jnp.log(mask + 1e-12) + k * idx / n
    weights = jax_softmax(log_score)
    return jnp.sum(weights * values)

def jax_softmax(x):
    m = jnp.max(x)
    e = jnp.exp(x - m)
    return e / jnp.sum(e)

def smooth_lookup_general(lookup_value, lookup_vector, result_vector, k=10.0):
    """
    Generic LOOKUP via soft attention.
    Weight on entry i = softmax_{i: v_i <= lookup_value}(k * v_i),
    smoothed so it's differentiable.
    """
    # Soft mask: entries where lookup_vector <= lookup_value
    gate = _sigmoid(lookup_value - lookup_vector, k)
    # Among those, prefer the largest → use lookup_vector as score
    log_score = jnp.log(gate + 1e-12) + k * lookup_vector
    weights = jax_softmax(log_score)
    return jnp.sum(weights * result_vector)

def smooth_offset_const(array, rows, cols, height=None, width=None):
    """
    OFFSET with constant (known-at-build-time) rows/cols is just an index.
    Variable offsets would need soft indexing; we handle the common case here.
    """
    base_idx = int(rows) * (width or 1) + int(cols)
    if height is None and width is None:
        return array[base_idx]
    h, w = int(height or 1), int(width or 1)
    return array[base_idx:base_idx + h * w].reshape(h, w)


# ---------------------------------------------------------------------------
# REGISTRY — for pycel and for the smoothing transformer
# ---------------------------------------------------------------------------
EXACT_REGISTRY = {
    'ABS': exact_abs, 'AND': exact_and, 'AVERAGE': exact_average,
    'IF': exact_if, 'IFERROR': exact_iferror, 'ISNUMBER': exact_isnumber,
    'LOOKUP': exact_lookup, 'NA': exact_na, 'NOT': exact_not,
    'OFFSET': exact_offset, 'OR': exact_or, 'PRODUCT': exact_product,
    'STDEV': exact_stdev, 'SUM': exact_sum, 'SUMPRODUCT': exact_sumproduct,
}

SMOOTH_REGISTRY = {
    'ABS': smooth_abs, 'AND': smooth_and, 'AVERAGE': smooth_average,
    'IF': smooth_if, 'IFERROR': smooth_iferror, 'ISNUMBER': smooth_isnumber,
    'NOT': smooth_not, 'OR': smooth_or, 'PRODUCT': smooth_product,
    'STDEV': smooth_stdev, 'SUM': smooth_sum, 'SUMPRODUCT': smooth_sumproduct,
    'LOOKUP': smooth_lookup_general, 'OFFSET': smooth_offset_const,
}
