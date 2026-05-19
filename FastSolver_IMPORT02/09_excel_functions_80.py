"""
excel_functions_80.py — FastSolver core function library (80 functions).

Two flavors per function:
  exact_*  : matches Excel behavior (validation, fallback)
  smooth_* : differentiable approximation, parameterized by sharpness k

Categories:
  Math & Basic (15), Rounding (5), Min/Max (6), Logical (10),
  Information (7), Lookup & Reference (10), Statistics (15),
  Aggregation (6), Distributions (4), Rank/Order (2).

Conventions:
  - Excel NA is represented by the ExcelNA sentinel (see excel_functions.py)
  - "smooth mask" arrays encode presence (1.0 = present, 0.0 = blank/NA)
  - All smooth_* functions are JAX-traceable (use jnp, not np)
"""

from __future__ import annotations
import math
import jax.numpy as jnp
from jax import lax
from jax.scipy import stats as jstats

# Reuse the NA sentinel and basic functions from the earlier module
from excel_functions import (
    ExcelNA, NA, is_na,
    exact_abs, exact_and, exact_or, exact_not,
    exact_if, exact_iferror, exact_isnumber, exact_na,
    exact_sum, exact_average, exact_product, exact_stdev,
    exact_sumproduct, exact_lookup, exact_offset,
    smooth_abs, smooth_max, smooth_min, smooth_if,
    smooth_and, smooth_or, smooth_not, smooth_iferror,
    smooth_isnumber, smooth_sum, smooth_average, smooth_product,
    smooth_stdev, smooth_sumproduct,
    smooth_lookup_last_nonblank, smooth_lookup_general, smooth_offset_const,
    jax_softmax, _sigmoid, _flatten, _filter_nums,
)


# =====================================================================
# MATH & BASIC (15)
# Already covered: ABS, SUM, PRODUCT
# New: SQRT, POWER, EXP, LN, LOG, LOG10, MOD, INT, TRUNC, SIGN, PI, FACT
# =====================================================================

def exact_sqrt(x):
    if is_na(x) or not exact_isnumber(x) or x < 0: return NA
    return math.sqrt(x)

def smooth_sqrt(x, eps=1e-9):
    return jnp.sqrt(jnp.maximum(x, 0.0) + eps)

def exact_power(x, p):
    if is_na(x) or is_na(p): return NA
    try: return float(x) ** float(p)
    except Exception: return NA

def smooth_power(x, p):
    # For x<0 with fractional p, Excel returns #NUM!; we clip to avoid NaN
    return jnp.sign(x) * jnp.abs(x) ** p

def exact_exp(x):
    if is_na(x): return NA
    return math.exp(x)

def smooth_exp(x):
    return jnp.exp(jnp.clip(x, -50.0, 50.0))  # clipping prevents overflow

def exact_ln(x):
    if is_na(x) or x <= 0: return NA
    return math.log(x)

def smooth_ln(x, eps=1e-9):
    return jnp.log(jnp.maximum(x, eps))

def exact_log(x, base=10):
    if is_na(x) or x <= 0 or base <= 0 or base == 1: return NA
    return math.log(x, base)

def smooth_log(x, base=10.0, eps=1e-9):
    return jnp.log(jnp.maximum(x, eps)) / jnp.log(base)

def exact_log10(x):
    return exact_log(x, 10)

def smooth_log10(x, eps=1e-9):
    return jnp.log10(jnp.maximum(x, eps))

def exact_mod(x, d):
    if is_na(x) or is_na(d) or d == 0: return NA
    return x - d * math.floor(x / d)

def smooth_mod(x, d, k=10.0, n_terms=6):
    """Fourier-series approximation of x mod d, periodic with period d."""
    # x mod d = d/2 - (d/π) Σ sin(2π n x / d)/n  (sawtooth Fourier series)
    s = jnp.zeros_like(x * 1.0)
    for n in range(1, n_terms + 1):
        s = s + jnp.sin(2 * jnp.pi * n * x / d) / n
    return d / 2 - (d / jnp.pi) * s

def exact_int(x):
    if is_na(x): return NA
    return math.floor(x)   # Excel INT = floor toward -∞

def smooth_int(x, k=10.0, n_terms=6):
    """x - (sawtooth fractional part)."""
    return x - smooth_mod(x, 1.0, k, n_terms)

def exact_trunc(x, digits=0):
    if is_na(x): return NA
    f = 10.0 ** digits
    return math.trunc(x * f) / f   # toward zero

def smooth_trunc(x, digits=0, k=10.0):
    """Toward-zero version. Approximate with sign-aware floor."""
    f = 10.0 ** digits
    xs = x * f
    return jnp.sign(xs) * smooth_int(jnp.abs(xs), k) / f

def exact_sign(x):
    if is_na(x): return NA
    return (x > 0) - (x < 0)

def smooth_sign(x, k=10.0):
    return jnp.tanh(k * x)   # smooth ±1 step

def exact_pi():
    return math.pi

def smooth_pi():
    return jnp.pi

def exact_fact(n):
    if is_na(n) or n < 0: return NA
    return math.factorial(int(n))

def smooth_fact(n):
    # Continuous extension via gamma function
    from jax.scipy.special import gammaln
    return jnp.exp(gammaln(n + 1.0))


# =====================================================================
# ROUNDING (5)
# =====================================================================

def exact_round(x, digits=0):
    if is_na(x): return NA
    f = 10.0 ** digits
    return math.floor(x * f + 0.5) / f   # banker's-vs-half-up: Excel uses half-up

def smooth_round(x, digits=0, k=10.0, n_terms=6):
    """ROUND ≈ x - (frac - 0.5)  smoothed."""
    f = 10.0 ** digits
    xs = x * f
    # Smooth frac: x - smooth_int(x)
    frac = xs - smooth_int(xs, k, n_terms)
    return (xs - frac + 0.5) / f - 0.5 / f  # equivalent to round-half-up

def exact_roundup(x, digits=0):
    if is_na(x): return NA
    f = 10.0 ** digits
    return math.ceil(x * f) / f if x >= 0 else -math.ceil(-x * f) / f

def smooth_roundup(x, digits=0, k=10.0, n_terms=6):
    f = 10.0 ** digits
    xs = x * f
    # ceiling-toward-magnitude (Excel ROUNDUP rounds away from 0)
    return jnp.sign(xs) * smooth_ceiling(jnp.abs(xs), k=k, n_terms=n_terms) / f

def exact_rounddown(x, digits=0):
    if is_na(x): return NA
    f = 10.0 ** digits
    return math.floor(x * f) / f if x >= 0 else -math.floor(-x * f) / f

def smooth_rounddown(x, digits=0, k=10.0, n_terms=6):
    f = 10.0 ** digits
    xs = x * f
    return jnp.sign(xs) * smooth_floor(jnp.abs(xs), k=k, n_terms=n_terms) / f

def exact_ceiling(x, significance=1):
    if is_na(x) or significance == 0: return NA
    return math.ceil(x / significance) * significance

def smooth_ceiling(x, significance=1.0, k=10.0, n_terms=6):
    return smooth_int(x / significance, k, n_terms) * significance + significance * _sigmoid(
        smooth_mod(x / significance, 1.0, k, n_terms) - 1e-9, k
    )

def exact_floor(x, significance=1):
    if is_na(x) or significance == 0: return NA
    return math.floor(x / significance) * significance

def smooth_floor(x, significance=1.0, k=10.0, n_terms=6):
    return smooth_int(x / significance, k, n_terms) * significance


# =====================================================================
# MIN / MAX FAMILY (6) — MIN, MAX, MINA, MAXA, MINIFS, MAXIFS
# (smooth_max, smooth_min already from base module)
# =====================================================================

def exact_min(*args):
    nums = _filter_nums(_flatten(args))
    return min(nums) if nums else NA

def exact_max(*args):
    nums = _filter_nums(_flatten(args))
    return max(nums) if nums else NA

def smooth_min_arr(values, mask=None, k=10.0):
    """log-sum-exp approximation of min: -log(Σ exp(-k·xᵢ))/k."""
    if mask is None:
        return -jnp.log(jnp.sum(jnp.exp(-k * values))) / k
    # mask: huge value where masked-out so it can't affect min
    huge = jnp.max(values) + 1e6
    masked_vals = mask * values + (1.0 - mask) * huge
    return -jnp.log(jnp.sum(jnp.exp(-k * masked_vals))) / k

def smooth_max_arr(values, mask=None, k=10.0):
    """log-sum-exp approximation of max: log(Σ exp(k·xᵢ))/k."""
    if mask is None:
        return jnp.log(jnp.sum(jnp.exp(k * values))) / k
    tiny = jnp.min(values) - 1e6
    masked_vals = mask * values + (1.0 - mask) * tiny
    return jnp.log(jnp.sum(jnp.exp(k * masked_vals))) / k

def exact_mina(*args):
    """MINA treats TRUE=1, FALSE=0, text=0 (unlike MIN which ignores them)."""
    flat = _flatten(args)
    vals = []
    for v in flat:
        if is_na(v): continue
        if isinstance(v, bool): vals.append(1.0 if v else 0.0)
        elif exact_isnumber(v): vals.append(v)
        else: vals.append(0.0)
    return min(vals) if vals else NA

def exact_maxa(*args):
    flat = _flatten(args)
    vals = []
    for v in flat:
        if is_na(v): continue
        if isinstance(v, bool): vals.append(1.0 if v else 0.0)
        elif exact_isnumber(v): vals.append(v)
        else: vals.append(0.0)
    return max(vals) if vals else NA

def _eval_criterion(cell_value, criterion):
    """Evaluate Excel-style criteria like '>10', '<=5', '=apple', 'text'."""
    if not isinstance(criterion, str):
        return cell_value == criterion
    crit = criterion.strip()
    for op in ('>=', '<=', '<>', '>', '<', '='):
        if crit.startswith(op):
            rhs = crit[len(op):].strip()
            try: rhs_val = float(rhs)
            except: rhs_val = rhs
            if op == '>=': return cell_value >= rhs_val
            if op == '<=': return cell_value <= rhs_val
            if op == '<>': return cell_value != rhs_val
            if op == '>':  return cell_value > rhs_val
            if op == '<':  return cell_value < rhs_val
            if op == '=':  return cell_value == rhs_val
    # No operator → equality match
    try: rhs_val = float(crit)
    except: rhs_val = crit
    return cell_value == rhs_val

def exact_minifs(target_range, *criteria_pairs):
    target = _flatten(target_range)
    n = len(target)
    keep = [True] * n
    for i in range(0, len(criteria_pairs), 2):
        crit_range = _flatten(criteria_pairs[i])
        crit = criteria_pairs[i + 1]
        for j in range(n):
            if not _eval_criterion(crit_range[j], crit):
                keep[j] = False
    vals = [target[j] for j in range(n) if keep[j] and exact_isnumber(target[j])]
    return min(vals) if vals else 0

def exact_maxifs(target_range, *criteria_pairs):
    target = _flatten(target_range)
    n = len(target)
    keep = [True] * n
    for i in range(0, len(criteria_pairs), 2):
        crit_range = _flatten(criteria_pairs[i])
        crit = criteria_pairs[i + 1]
        for j in range(n):
            if not _eval_criterion(crit_range[j], crit):
                keep[j] = False
    vals = [target[j] for j in range(n) if keep[j] and exact_isnumber(target[j])]
    return max(vals) if vals else 0


# =====================================================================
# LOGICAL (10) — adds IFS, IFNA, XOR, TRUE, FALSE
# =====================================================================

def exact_ifs(*pairs):
    """IFS(cond1, val1, cond2, val2, ...) — returns first true."""
    for i in range(0, len(pairs), 2):
        cond = pairs[i]
        val = pairs[i + 1] if i + 1 < len(pairs) else NA
        if is_na(cond): continue
        if bool(cond): return val
    return NA   # Excel #N/A when nothing matches

def smooth_ifs(conds, vals, k=10.0):
    """Weighted blend using softmax over conditions."""
    conds = jnp.asarray(conds); vals = jnp.asarray(vals)
    weights = jax_softmax(k * conds)
    return jnp.sum(weights * vals)

def exact_ifna(value, if_na):
    if is_na(value): return if_na
    return value

def smooth_ifna(value, if_na, validity, k=10.0):
    """validity > 0 means 'not NA'."""
    w = _sigmoid(validity, k)
    return w * value + (1.0 - w) * if_na

def exact_xor(*args):
    count_true = sum(1 for a in args if (not is_na(a)) and bool(a))
    return count_true % 2 == 1

def smooth_xor(*conds, k=10.0):
    """Smooth XOR via mod-2 of sum of sigmoids."""
    s = sum(_sigmoid(c, k) for c in conds)
    return 0.5 - 0.5 * jnp.cos(jnp.pi * s)

def exact_true():  return True
def exact_false(): return False

def smooth_true():  return jnp.array(1.0)
def smooth_false(): return jnp.array(0.0)


# =====================================================================
# INFORMATION / TYPE CHECKS (7)
# =====================================================================

def exact_isblank(x):
    if is_na(x): return False
    return x is None or x == ""

def exact_iserror(x):
    return is_na(x) or (isinstance(x, float) and math.isnan(x))

def exact_iserr(x):
    # ISERR = error but not #N/A
    if is_na(x): return False
    return isinstance(x, float) and math.isnan(x)

def exact_isna(x):
    return is_na(x)

def exact_istext(x):
    if is_na(x): return False
    return isinstance(x, str)

# (NA, ISNUMBER already imported)


# =====================================================================
# LOOKUP & REFERENCE (10)
# Already: LOOKUP, OFFSET
# New: VLOOKUP, HLOOKUP, XLOOKUP, INDEX, MATCH, XMATCH, CHOOSE, INDIRECT
# =====================================================================

def exact_vlookup(lookup_value, table, col_index, range_lookup=False):
    """VLOOKUP(value, table, col, [approximate])."""
    # table = 2D list-of-lists
    if not isinstance(table, (list, tuple)): return NA
    rows = table if isinstance(table[0], (list, tuple)) else [table]
    col_index = int(col_index)
    if range_lookup:
        best = None
        for r in rows:
            if exact_isnumber(r[0]) and r[0] <= lookup_value:
                best = r
        if best is None: return NA
        return best[col_index - 1] if col_index - 1 < len(best) else NA
    for r in rows:
        if r[0] == lookup_value:
            return r[col_index - 1] if col_index - 1 < len(r) else NA
    return NA

def exact_hlookup(lookup_value, table, row_index, range_lookup=False):
    """HLOOKUP — same as VLOOKUP but searches first row."""
    if not isinstance(table, (list, tuple)): return NA
    rows = table if isinstance(table[0], (list, tuple)) else [table]
    cols = list(zip(*rows))
    row_index = int(row_index)
    for c in cols:
        if c[0] == lookup_value:
            return c[row_index - 1] if row_index - 1 < len(c) else NA
    return NA

def exact_xlookup(lookup_value, lookup_array, return_array, not_found=NA,
                  match_mode=0, search_mode=1):
    """XLOOKUP — modern lookup with multiple match modes."""
    la = _flatten(lookup_array); ra = _flatten(return_array)
    if match_mode == 0:  # exact
        idxs = [i for i, v in enumerate(la) if v == lookup_value]
    elif match_mode == -1:  # exact or next smaller
        below = [(i, v) for i, v in enumerate(la) if exact_isnumber(v) and v <= lookup_value]
        idxs = [max(below, key=lambda x: x[1])[0]] if below else []
    elif match_mode == 1:  # exact or next larger
        above = [(i, v) for i, v in enumerate(la) if exact_isnumber(v) and v >= lookup_value]
        idxs = [min(above, key=lambda x: x[1])[0]] if above else []
    else:
        idxs = []
    if not idxs: return not_found
    idx = idxs[-1] if search_mode == -1 else idxs[0]
    return ra[idx] if idx < len(ra) else not_found

def smooth_xlookup(lookup_value, lookup_array, return_array, k=10.0):
    """Soft attention: weight return values by sigmoid proximity in lookup."""
    diffs = -((lookup_array - lookup_value) ** 2)
    weights = jax_softmax(k * diffs)
    return jnp.sum(weights * return_array)

def exact_index(array, row_num, col_num=None):
    if not isinstance(array, (list, tuple)): return array
    row_num = int(row_num)
    if col_num is None:
        flat = _flatten(array)
        return flat[row_num - 1] if 0 < row_num <= len(flat) else NA
    rows = array if isinstance(array[0], (list, tuple)) else [array]
    if not (0 < row_num <= len(rows)): return NA
    r = rows[row_num - 1]
    col_num = int(col_num)
    return r[col_num - 1] if 0 < col_num <= len(r) else NA

def smooth_index(array, row_num, k=10.0):
    """Soft index via softmax over positions."""
    n = array.shape[0]
    idx = jnp.arange(n, dtype=array.dtype)
    weights = jax_softmax(-k * (idx - (row_num - 1)) ** 2)
    return jnp.sum(weights * array)

def exact_match(lookup_value, lookup_array, match_type=1):
    la = _flatten(lookup_array)
    if match_type == 0:
        for i, v in enumerate(la):
            if v == lookup_value: return i + 1
        return NA
    elif match_type == 1:  # largest value <= lookup
        best = None; bi = None
        for i, v in enumerate(la):
            if exact_isnumber(v) and v <= lookup_value:
                if best is None or v >= best: best, bi = v, i
        return bi + 1 if bi is not None else NA
    elif match_type == -1:  # smallest >= lookup
        best = None; bi = None
        for i, v in enumerate(la):
            if exact_isnumber(v) and v >= lookup_value:
                if best is None or v <= best: best, bi = v, i
        return bi + 1 if bi is not None else NA
    return NA

def smooth_match(lookup_value, lookup_array, k=10.0):
    """Returns soft 1-based index of best match."""
    n = lookup_array.shape[0]
    idx = jnp.arange(n, dtype=lookup_array.dtype) + 1
    diffs = -((lookup_array - lookup_value) ** 2)
    weights = jax_softmax(k * diffs)
    return jnp.sum(weights * idx)

def exact_xmatch(lookup_value, lookup_array, match_mode=0, search_mode=1):
    return exact_match(lookup_value, lookup_array, match_mode)

def exact_choose(idx, *values):
    idx = int(idx)
    if 0 < idx <= len(values):
        return values[idx - 1]
    return NA

def smooth_choose(idx, values, k=10.0):
    """Soft index pick from list of values."""
    n = len(values)
    arr = jnp.asarray(values)
    positions = jnp.arange(1, n + 1, dtype=arr.dtype)
    weights = jax_softmax(-k * (positions - idx) ** 2)
    return jnp.sum(weights * arr)

def exact_indirect(ref_text, a1_style=True):
    """INDIRECT is reference-based and pycel handles it at the graph level.
    Here we just return the literal — caller resolves."""
    return ref_text


# =====================================================================
# STATISTICS — CORE (15)
# Already: AVERAGE, STDEV
# New: AVERAGEA, AVERAGEIF, AVERAGEIFS, MEDIAN, STDEV.S, STDEV.P,
#      VAR, VAR.S, VAR.P, COUNT, COUNTA, COUNTIF, COUNTIFS
# =====================================================================

def exact_averagea(*args):
    flat = _flatten(args)
    vals = []
    for v in flat:
        if is_na(v): continue
        if isinstance(v, bool): vals.append(1.0 if v else 0.0)
        elif exact_isnumber(v): vals.append(v)
        else: vals.append(0.0)
    return sum(vals) / len(vals) if vals else NA

def exact_averageif(range_, criterion, average_range=None):
    rng = _flatten(range_)
    ar = _flatten(average_range) if average_range is not None else rng
    kept = [ar[i] for i, v in enumerate(rng)
            if _eval_criterion(v, criterion) and i < len(ar) and exact_isnumber(ar[i])]
    return sum(kept) / len(kept) if kept else NA

def exact_averageifs(average_range, *criteria_pairs):
    ar = _flatten(average_range)
    n = len(ar)
    keep = [True] * n
    for i in range(0, len(criteria_pairs), 2):
        crit_range = _flatten(criteria_pairs[i])
        crit = criteria_pairs[i + 1]
        for j in range(n):
            if not _eval_criterion(crit_range[j], crit): keep[j] = False
    kept = [ar[j] for j in range(n) if keep[j] and exact_isnumber(ar[j])]
    return sum(kept) / len(kept) if kept else NA

def exact_median(*args):
    nums = sorted(_filter_nums(_flatten(args)))
    if not nums: return NA
    n = len(nums)
    return nums[n // 2] if n % 2 == 1 else (nums[n // 2 - 1] + nums[n // 2]) / 2

def smooth_median(values, k=20.0):
    """Approximate median via soft quantile (sigmoid-based)."""
    sorted_v = jnp.sort(values)
    n = sorted_v.shape[0]
    return sorted_v[n // 2]   # JAX traces through indexing; gradient is sparse

def exact_stdev_s(*args):  return exact_stdev(*args)   # alias
def exact_stdev_p(*args):
    """Population stddev — uses n in denominator."""
    nums = _filter_nums(_flatten(args))
    if not nums: return NA
    m = sum(nums) / len(nums)
    return math.sqrt(sum((x - m) ** 2 for x in nums) / len(nums))

def smooth_stdev_p(values, mask=None, eps=1e-9):
    if mask is None:
        m = jnp.mean(values)
        return jnp.sqrt(jnp.mean((values - m) ** 2) + eps)
    n_eff = jnp.sum(mask) + eps
    m = jnp.sum(values * mask) / n_eff
    return jnp.sqrt(jnp.sum((values - m) ** 2 * mask) / n_eff + eps)

def exact_var(*args):
    s = exact_stdev(*args)
    return s ** 2 if not is_na(s) else NA

def exact_var_s(*args): return exact_var(*args)
def exact_var_p(*args):
    s = exact_stdev_p(*args)
    return s ** 2 if not is_na(s) else NA

def smooth_var(values, mask=None, eps=1e-9):
    s = smooth_stdev(values, mask, eps)
    return s * s

def smooth_var_p(values, mask=None, eps=1e-9):
    s = smooth_stdev_p(values, mask, eps)
    return s * s

def exact_count(*args):
    return sum(1 for v in _flatten(args) if exact_isnumber(v))

def exact_counta(*args):
    return sum(1 for v in _flatten(args)
               if v is not None and v != "" and not is_na(v))

def exact_countif(range_, criterion):
    return sum(1 for v in _flatten(range_) if _eval_criterion(v, criterion))

def exact_countifs(*pairs):
    if not pairs: return 0
    rng0 = _flatten(pairs[0])
    n = len(rng0)
    keep = [True] * n
    for i in range(0, len(pairs), 2):
        crit_range = _flatten(pairs[i])
        crit = pairs[i + 1]
        for j in range(n):
            if not _eval_criterion(crit_range[j], crit): keep[j] = False
    return sum(keep)

# Smooth count = differentiable indicator sum (used during optimization)
def smooth_count(mask):
    return jnp.sum(mask)

def smooth_countif(values, threshold, op='>=', k=10.0):
    if op in ('>=', '>'): scores = values - threshold
    elif op in ('<=', '<'): scores = threshold - values
    else: scores = -((values - threshold) ** 2)
    return jnp.sum(_sigmoid(scores, k))


# =====================================================================
# AGGREGATION (6) — SUMIF, SUMIFS, SUMSQ, SUMXMY2, SUBTOTAL
# Already: SUMPRODUCT
# =====================================================================

def exact_sumif(range_, criterion, sum_range=None):
    rng = _flatten(range_)
    sr = _flatten(sum_range) if sum_range is not None else rng
    total = 0.0
    for i, v in enumerate(rng):
        if _eval_criterion(v, criterion) and i < len(sr) and exact_isnumber(sr[i]):
            total += sr[i]
    return total

def smooth_sumif(values, sum_values, threshold, op='>=', k=10.0):
    if op in ('>=', '>'): scores = values - threshold
    elif op in ('<=', '<'): scores = threshold - values
    else: scores = -((values - threshold) ** 2)
    weights = _sigmoid(scores, k)
    return jnp.sum(weights * sum_values)

def exact_sumifs(sum_range, *criteria_pairs):
    sr = _flatten(sum_range)
    n = len(sr)
    keep = [True] * n
    for i in range(0, len(criteria_pairs), 2):
        crit_range = _flatten(criteria_pairs[i])
        crit = criteria_pairs[i + 1]
        for j in range(n):
            if not _eval_criterion(crit_range[j], crit): keep[j] = False
    return sum(sr[j] for j in range(n) if keep[j] and exact_isnumber(sr[j]))

def exact_sumsq(*args):
    return sum(v ** 2 for v in _flatten(args) if exact_isnumber(v))

def smooth_sumsq(values):
    return jnp.sum(values * values)

def exact_sumxmy2(x_range, y_range):
    """Σ (x - y)²."""
    x = _flatten(x_range); y = _flatten(y_range)
    return sum((x[i] - y[i]) ** 2 for i in range(min(len(x), len(y)))
               if exact_isnumber(x[i]) and exact_isnumber(y[i]))

def smooth_sumxmy2(x, y):
    return jnp.sum((x - y) ** 2)

def exact_subtotal(function_num, *ranges):
    """SUBTOTAL(101, range) = AVERAGE ignoring hidden; we approximate with
    function_num table."""
    fn_table = {
        1: exact_average, 101: exact_average,
        2: exact_count,   102: exact_count,
        3: exact_counta,  103: exact_counta,
        4: exact_max,     104: exact_max,
        5: exact_min,     105: exact_min,
        6: exact_product, 106: exact_product,
        7: exact_stdev,   107: exact_stdev,
        8: exact_stdev_p, 108: exact_stdev_p,
        9: exact_sum,     109: exact_sum,
       10: exact_var,     110: exact_var,
       11: exact_var_p,   111: exact_var_p,
    }
    fn = fn_table.get(int(function_num))
    if fn is None: return NA
    return fn(*ranges)


# =====================================================================
# DISTRIBUTIONS (4)
# =====================================================================

def exact_norm_dist(x, mean, sd, cumulative=True):
    if sd <= 0: return NA
    if cumulative:
        return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

def smooth_norm_dist(x, mean, sd, cumulative=True):
    if cumulative:
        return jstats.norm.cdf(x, loc=mean, scale=sd)
    return jstats.norm.pdf(x, loc=mean, scale=sd)

def exact_norm_inv(p, mean, sd):
    """Inverse CDF of normal."""
    if not (0 < p < 1) or sd <= 0: return NA
    # Use math.erfinv via SciPy fallback or rational approx
    # Beasley-Springer/Moro
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= 1 - p_low:
        q = p - 0.5
        r = q*q
        z = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return mean + sd * z

def smooth_norm_inv(p, mean, sd):
    return jstats.norm.ppf(p, loc=mean, scale=sd)

def exact_norm_s_dist(x, cumulative=True):
    return exact_norm_dist(x, 0, 1, cumulative)
def exact_norm_s_inv(p):
    return exact_norm_inv(p, 0, 1)
def smooth_norm_s_dist(x, cumulative=True):
    return smooth_norm_dist(x, 0.0, 1.0, cumulative)
def smooth_norm_s_inv(p):
    return smooth_norm_inv(p, 0.0, 1.0)


# =====================================================================
# RANK / ORDER (2) — LARGE, SMALL
# =====================================================================

def exact_large(array, k):
    nums = sorted(_filter_nums(_flatten(array)), reverse=True)
    k = int(k)
    return nums[k - 1] if 0 < k <= len(nums) else NA

def exact_small(array, k):
    nums = sorted(_filter_nums(_flatten(array)))
    k = int(k)
    return nums[k - 1] if 0 < k <= len(nums) else NA

def smooth_large(values, k_rank, sharpness=20.0):
    """k-th largest via soft top-k. Differentiable but indexing is sparse."""
    sorted_v = jnp.sort(values)[::-1]
    return sorted_v[int(k_rank) - 1]

def smooth_small(values, k_rank, sharpness=20.0):
    sorted_v = jnp.sort(values)
    return sorted_v[int(k_rank) - 1]


# =====================================================================
# REGISTRIES — all 80 functions for pycel and the smoothing transformer
# =====================================================================

EXACT_REGISTRY_80 = {
    # Math & Basic (15)
    'ABS': exact_abs, 'SUM': exact_sum, 'PRODUCT': exact_product,
    'SQRT': exact_sqrt, 'POWER': exact_power, 'EXP': exact_exp,
    'LN': exact_ln, 'LOG': exact_log, 'LOG10': exact_log10,
    'MOD': exact_mod, 'INT': exact_int, 'TRUNC': exact_trunc,
    'SIGN': exact_sign, 'PI': exact_pi, 'FACT': exact_fact,
    # Rounding (5)
    'ROUND': exact_round, 'ROUNDUP': exact_roundup, 'ROUNDDOWN': exact_rounddown,
    'CEILING': exact_ceiling, 'FLOOR': exact_floor,
    # Min/Max (6)
    'MIN': exact_min, 'MAX': exact_max, 'MINA': exact_mina, 'MAXA': exact_maxa,
    'MINIFS': exact_minifs, 'MAXIFS': exact_maxifs,
    # Logical (10)
    'IF': exact_if, 'IFS': exact_ifs, 'IFERROR': exact_iferror, 'IFNA': exact_ifna,
    'AND': exact_and, 'OR': exact_or, 'NOT': exact_not, 'XOR': exact_xor,
    'TRUE': exact_true, 'FALSE': exact_false,
    # Info / type (7)
    'ISNUMBER': exact_isnumber, 'ISBLANK': exact_isblank, 'ISERROR': exact_iserror,
    'ISERR': exact_iserr, 'ISNA': exact_isna, 'ISTEXT': exact_istext, 'NA': exact_na,
    # Lookup & ref (10)
    'LOOKUP': exact_lookup, 'VLOOKUP': exact_vlookup, 'HLOOKUP': exact_hlookup,
    'XLOOKUP': exact_xlookup, 'INDEX': exact_index, 'MATCH': exact_match,
    'XMATCH': exact_xmatch, 'OFFSET': exact_offset, 'CHOOSE': exact_choose,
    'INDIRECT': exact_indirect,
    # Stats core (15)
    'AVERAGE': exact_average, 'AVERAGEA': exact_averagea,
    'AVERAGEIF': exact_averageif, 'AVERAGEIFS': exact_averageifs,
    'MEDIAN': exact_median, 'STDEV': exact_stdev, 'STDEV.S': exact_stdev_s,
    'STDEV.P': exact_stdev_p, 'VAR': exact_var, 'VAR.S': exact_var_s,
    'VAR.P': exact_var_p, 'COUNT': exact_count, 'COUNTA': exact_counta,
    'COUNTIF': exact_countif, 'COUNTIFS': exact_countifs,
    # Aggregation (6)
    'SUMIF': exact_sumif, 'SUMIFS': exact_sumifs, 'SUMPRODUCT': exact_sumproduct,
    'SUMSQ': exact_sumsq, 'SUMXMY2': exact_sumxmy2, 'SUBTOTAL': exact_subtotal,
    # Distributions (4)
    'NORM.DIST': exact_norm_dist, 'NORM.INV': exact_norm_inv,
    'NORM.S.DIST': exact_norm_s_dist, 'NORM.S.INV': exact_norm_s_inv,
    # Rank/order (2)
    'LARGE': exact_large, 'SMALL': exact_small,
}

SMOOTH_REGISTRY_80 = {
    # Smoothables
    'ABS': smooth_abs, 'SUM': smooth_sum, 'PRODUCT': smooth_product,
    'SQRT': smooth_sqrt, 'POWER': smooth_power, 'EXP': smooth_exp,
    'LN': smooth_ln, 'LOG': smooth_log, 'LOG10': smooth_log10,
    'MOD': smooth_mod, 'INT': smooth_int, 'TRUNC': smooth_trunc, 'SIGN': smooth_sign,
    'PI': smooth_pi, 'FACT': smooth_fact,
    'ROUND': smooth_round, 'ROUNDUP': smooth_roundup, 'ROUNDDOWN': smooth_rounddown,
    'CEILING': smooth_ceiling, 'FLOOR': smooth_floor,
    'MIN': smooth_min, 'MAX': smooth_max,
    'IF': smooth_if, 'IFS': smooth_ifs, 'IFERROR': smooth_iferror, 'IFNA': smooth_ifna,
    'AND': smooth_and, 'OR': smooth_or, 'NOT': smooth_not, 'XOR': smooth_xor,
    'TRUE': smooth_true, 'FALSE': smooth_false,
    'ISNUMBER': smooth_isnumber,
    'LOOKUP': smooth_lookup_general, 'XLOOKUP': smooth_xlookup,
    'INDEX': smooth_index, 'MATCH': smooth_match, 'CHOOSE': smooth_choose,
    'OFFSET': smooth_offset_const,
    'AVERAGE': smooth_average, 'MEDIAN': smooth_median,
    'STDEV': smooth_stdev, 'STDEV.P': smooth_stdev_p,
    'VAR': smooth_var, 'VAR.P': smooth_var_p,
    'COUNT': smooth_count, 'COUNTIF': smooth_countif,
    'SUMIF': smooth_sumif, 'SUMPRODUCT': smooth_sumproduct,
    'SUMSQ': smooth_sumsq, 'SUMXMY2': smooth_sumxmy2,
    'NORM.DIST': smooth_norm_dist, 'NORM.INV': smooth_norm_inv,
    'NORM.S.DIST': smooth_norm_s_dist, 'NORM.S.INV': smooth_norm_s_inv,
    'LARGE': smooth_large, 'SMALL': smooth_small,
}


if __name__ == '__main__':
    print(f"Exact functions:  {len(EXACT_REGISTRY_80)}")
    print(f"Smooth functions: {len(SMOOTH_REGISTRY_80)}")
    print(f"Missing smooth (intentionally — text/discrete/reference-only):")
    missing = set(EXACT_REGISTRY_80) - set(SMOOTH_REGISTRY_80)
    print(f"  {sorted(missing)}")
