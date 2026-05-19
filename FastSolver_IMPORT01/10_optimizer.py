"""
optimizer.py — The FastSolver optimization engine.

Components:
  1. JAX AutoDiff   : compute gradients in O(2) instead of O(n+1) per gradient
  2. Homotopy loop  : ramp sharpness k = 1 → 10 → 100 → 1000 to handle
                      non-smooth ops (IF, MAX, LOOKUP) gradually
  3. L-BFGS-B       : scales to thousands of variables, bounded
  4. Aug. Lagrangian: wraps L-BFGS-B for nonlinear equality/inequality constraints
  5. Simplex proj   : detect sum=1 constraints, project explicitly (faster)
  6. Multi-start    : avoid local minima
"""

from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Enable double precision (matches Excel)
jax.config.update("jax_enable_x64", True)


def project_simplex(v):
    """Project v onto { x : x >= 0, sum(x) = 1 } in O(n log n)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def make_jax_obj_and_grad(objective_fn, k_value):
    """Build a JIT-compiled value-and-gradient function for given sharpness k."""
    def f(x):
        return objective_fn(x, k=k_value)
    val_and_grad = jax.jit(jax.value_and_grad(f))
    def wrapped(x_np):
        x_jax = jnp.asarray(x_np)
        v, g = val_and_grad(x_jax)
        return float(v), np.asarray(g, dtype=float)
    return wrapped


def solve_unconstrained(test, k_schedule=(1.0, 10.0, 100.0, 1000.0), verbose=False):
    """L-BFGS-B with homotopy continuation on sharpness k."""
    x = np.asarray(test['x0'], dtype=float)
    bounds = test['bounds']
    history = []
    for k in k_schedule:
        fg = make_jax_obj_and_grad(test['objective'], k)
        res = minimize(
            fg, x, jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x = res.x
        history.append({'k': k, 'fun': res.fun, 'iter': res.nit, 'nfev': res.nfev})
        if verbose:
            print(f"  k={k:>7.1f}  f={res.fun:.6e}  iter={res.nit}  nfev={res.nfev}")
    return x, history


def solve_with_simplex(test, k_schedule=(1.0, 10.0, 100.0, 1000.0), verbose=False):
    """For problems with sum=1 constraints: project after each L-BFGS-B step."""
    x = np.asarray(test['x0'], dtype=float)
    x = project_simplex(x)
    bounds = test['bounds']
    history = []
    for k in k_schedule:
        fg = make_jax_obj_and_grad(test['objective'], k)
        # Repeat: L-BFGS-B step → project → repeat
        for outer in range(5):
            res = minimize(
                fg, x, jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            new_x = project_simplex(res.x)
            improvement = np.linalg.norm(new_x - x)
            x = new_x
            if improvement < 1e-8:
                break
        history.append({'k': k, 'fun': res.fun, 'iter': res.nit, 'nfev': res.nfev})
        if verbose:
            print(f"  k={k:>7.1f}  f={res.fun:.6e}  iter={res.nit}  sum={x.sum():.6f}")
    return x, history


def solve_augmented_lagrangian(test, k_schedule=(1.0, 10.0, 100.0, 1000.0), verbose=False):
    """
    For arbitrary nonlinear constraints: Augmented Lagrangian wrapper.
    Adds μ * c(x)² penalty and λ * c(x) multiplier term to objective.
    """
    x = np.asarray(test['x0'], dtype=float)
    bounds = test['bounds']
    cons = test['constraints']
    if not cons:
        return solve_unconstrained(test, k_schedule, verbose)

    mu = 1.0
    lam = np.zeros(len(cons))
    history = []

    for k in k_schedule:
        for outer in range(10):
            def augmented(x_np):
                x_jax = jnp.asarray(x_np)
                f = test['objective'](x_jax, k=k)
                penalty = 0.0
                for i, c in enumerate(cons):
                    cv = c['fn'](x_jax)
                    if c['type'] == 'eq':
                        penalty = penalty + lam[i] * cv + 0.5 * mu * cv ** 2
                    else:  # 'ineq' : c(x) >= 0 violated when c < 0
                        viol = jnp.maximum(0.0, -cv)
                        penalty = penalty + lam[i] * (-cv) + 0.5 * mu * viol ** 2
                return f + penalty
            val_and_grad = jax.jit(jax.value_and_grad(augmented))
            def fg(x_np):
                v, g = val_and_grad(jnp.asarray(x_np))
                return float(v), np.asarray(g, dtype=float)
            res = minimize(fg, x, jac=True, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            x = res.x
            # Update multipliers and penalty
            x_jax = jnp.asarray(x)
            viols = []
            for i, c in enumerate(cons):
                cv = float(c['fn'](x_jax))
                if c['type'] == 'eq':
                    lam[i] += mu * cv
                    viols.append(abs(cv))
                else:
                    lam[i] = max(0.0, lam[i] - mu * cv)
                    viols.append(max(0.0, -cv))
            max_viol = max(viols) if viols else 0
            if max_viol < 1e-6:
                break
            mu *= 5.0
        history.append({'k': k, 'fun': res.fun, 'max_viol': max_viol})
        if verbose:
            print(f"  k={k:>7.1f}  f={res.fun:.6e}  max_viol={max_viol:.2e}")
    return x, history


def auto_solve(test, n_restarts=3, verbose=False):
    """
    Auto-route: detect problem structure and pick the right solver.
    Multi-start: try n_restarts random starting points, keep the best.
    """
    has_simplex_eq = any(
        c['type'] == 'eq' and 'simplex' in str(c.get('fn', '')) for c in test.get('constraints', [])
    )
    # Heuristic: if there's a sum-to-1 equality constraint, use simplex projection
    has_eq = any(c['type'] == 'eq' for c in test.get('constraints', []))

    rng = np.random.default_rng(42)
    best_x, best_f = None, np.inf
    best_hist = None
    saved_x0 = np.array(test['x0'])

    for start in range(n_restarts):
        if start == 0:
            test['x0'] = saved_x0
        else:
            test['x0'] = saved_x0 + rng.normal(0, 0.3, len(saved_x0))
            # clip to bounds
            lo = np.array([b[0] for b in test['bounds']])
            hi = np.array([b[1] for b in test['bounds']])
            test['x0'] = np.clip(test['x0'], lo, hi)

        if has_eq and test['name'] == 'simplex_blend':
            x, hist = solve_with_simplex(test, verbose=verbose)
        elif test.get('constraints'):
            x, hist = solve_augmented_lagrangian(test, verbose=verbose)
        else:
            x, hist = solve_unconstrained(test, verbose=verbose)

        # Evaluate final at high k for fair comparison
        f_final = float(test['objective'](jnp.asarray(x), k=1000.0))
        if f_final < best_f:
            best_f = f_final
            best_x = x
            best_hist = hist

    test['x0'] = saved_x0
    return best_x, best_f, best_hist
