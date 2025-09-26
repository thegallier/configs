import jax
import jax.numpy as jnp

def schedule_with_stub(T, freq=2, dtype=jnp.float64):
    f = jnp.array(freq, dtype=dtype)
    n_full = jnp.floor(T * f).astype(jnp.int32)
    idx    = jnp.arange(1, n_full + 1, dtype=jnp.int32)
    t_full = idx.astype(dtype) / f
    a_full = jnp.full_like(t_full, 1.0 / f, dtype=dtype)
    d_stub = T - (n_full.astype(dtype) / f)
    has_stub = d_stub > jnp.array(1e-14, dtype=dtype)
    def _with_stub(t, a):
        return (jnp.concatenate([t, T[None]]),
                jnp.concatenate([a, d_stub[None]]))
    def _no_stub(t, a):
        return t, a
    return jax.lax.cond(has_stub, _with_stub, _no_stub, t_full, a_full)

def bond_price_continuous(T, c, discount_fn, freq=2, dtype=jnp.float64):
    """
    Price of a coupon bond with annual coupon rate c, maturity T, frequency freq,
    using a final stub coupon if T is off-schedule. discount_fn(t) must return D(t).
    """
    times, accruals = schedule_with_stub(T, freq=freq, dtype=dtype)
    D_times = jax.vmap(discount_fn)(times)
    # Principal at T plus stub coupon (already included via accruals at T):
    # sum full coupons + stub coupon + principal
    coupons = jnp.sum(c * accruals * D_times)
    principal = discount_fn(T)
    return coupons + principal


import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

DTYPE = jnp.float64

# ---- Your existing building blocks (unchanged) ----
def safe_exp(x): return jnp.exp(jnp.clip(x, -80.0, 40.0))

def integral_coeffs(T, A):
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    expAT = expm(A * T)
    Ainv  = jnp.linalg.pinv(A)
    Ainv2 = Ainv @ Ainv
    R1 = Ainv @ (expAT - I)
    R2 = Ainv2 @ (expAT - I - A * T)
    return R1, R2

def log_discount_det(T, A, X, d, p):
    R1, R2 = integral_coeffs(T, A)
    return -(p @ (R1 @ X + R2 @ d))

def discount_det(T, A, X, d, p):
    return safe_exp(log_discount_det(T, A, X, d, p))

# Optional: your exact Gaussian convexity for ∫ z ds (if you already had it)
def var_integrated_z_exact(T, A, Sigma, p):
    # Augmented 2n×2n method; omit here for brevity or plug your existing function
    n = A.shape[0]
    Az = jnp.block([[A,                          jnp.zeros((n,n), A.dtype)],
                    [jnp.eye(n, dtype=A.dtype),  jnp.zeros((n,n), A.dtype)]])
    Bz  = jnp.vstack([Sigma, jnp.zeros((n, Sigma.shape[1]), A.dtype)])
    BBt = Bz @ Bz.T
    I2n = jnp.eye(2*n, dtype=A.dtype)
    L   = jnp.kron(I2n, Az) + jnp.kron(Az, I2n)
    expLT = expm(L * T)
    rhs   = (expLT - jnp.eye(L.shape[0], dtype=A.dtype)) @ BBt.reshape(-1)
    vecC, *_ = jnp.linalg.lstsq(L, rhs, rcond=None)
    Cov = vecC.reshape(2*n, 2*n)
    CovYY = Cov[n:, n:]
    return p @ CovYY @ p  # Var[∫_0^T z ds]

# ---- NEW: schedule with stub accrual ----
def schedule_with_stub(T, freq=2, dtype=DTYPE):
    """Return coupon times and accruals including a final stub at T."""
    f = jnp.array(freq, dtype=dtype)
    n_full = jnp.floor(T * f).astype(jnp.int32)           # number of full coupons
    # full coupons (possibly zero-length)
    idx_full = jnp.arange(1, n_full + 1, dtype=jnp.int32)
    t_full   = idx_full.astype(dtype) / f
    d_full   = jnp.full_like(t_full, 1.0 / f, dtype=dtype)
    # stub
    t_last   = T
    d_stub   = T - (n_full.astype(dtype) / f)             # in years
    has_stub = d_stub > jnp.array(1e-14, dtype=dtype)

    # Concatenate safely even when n_full==0 or no stub
    def _cat_stub(t_full, d_full):
        t_all = jnp.concatenate([t_full, t_last[None]], axis=0)
        d_all = jnp.concatenate([d_full, d_stub[None]], axis=0)
        return t_all, d_all
    def _no_stub(t_full, d_full):
        return t_full, d_full

    t_all, d_all = jax.lax.cond(has_stub, _cat_stub, _no_stub, t_full, d_full)
    return t_all, d_all

# ---- Continuous par-rate (with optional convexity) ----
def par_rate_continuous(T, A, X, d, p, Sigma=None, include_var=False, freq=2):
    times, accruals = schedule_with_stub(T, freq=freq, dtype=A.dtype)

    if include_var and (Sigma is not None):
        # P(t) = exp( -M(t) + 0.5 Var[∫_0^t z ds] )
        def P_t(t):
            logP = log_discount_det(t, A, X, d, p)
            V    = var_integrated_z_exact(t, A, Sigma, p)
            return safe_exp(logP + 0.5 * V)
        P_vec = jax.vmap(P_t)(times)
        P_T   = P_t(T)
    else:
        P_vec = jax.vmap(lambda t: discount_det(t, A, X, d, p))(times)
        P_T   = discount_det(T, A, X, d, p)

    annuity = jnp.sum(accruals * P_vec)
    annuity = jnp.maximum(annuity, jnp.array(1e-14, dtype=A.dtype))
    return (1.0 - P_T) / annuity

# Vectorized curve
def par_curve_continuous(mats, A, X, d, p, Sigma=None, include_var=False, freq=2):
    return jax.vmap(lambda T: par_rate_continuous(T, A, X, d, p, Sigma, include_var, freq))(mats)
