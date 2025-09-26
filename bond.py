import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

DTYPE = jnp.float64

def safe_exp(x):
    return jnp.exp(jnp.clip(x, -80.0, 40.0))

# -------------------------------
# Discount from OU with convexity
# -------------------------------
from scipy.linalg import expm as scipy_expm

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, pinv

def integral_coeffs(T, A):
    I     = np.eye(A.shape[0], dtype=np.float64)
    expAT = expm(np.asarray(A) * float(T))        # SciPy expm
    Ainv  = pinv(np.asarray(A))                   # SciPy pinv
    Ainv2 = Ainv @ Ainv
    R1 = Ainv @ (expAT - I)
    R2 = Ainv2 @ (expAT - I - np.asarray(A) * float(T))
    return jnp.array(R1), jnp.array(R2)           # back to JAX arrays

def log_discount_det(T, A, X0, d, p):
    R1, R2 = integral_coeffs(T, A)
    M = p @ (R1 @ X0 + R2 @ d)
    return -M

def var_integrated_z_exact(T, A, Sigma, p):
    n = A.shape[0]
    Az = jnp.block([
        [A,                         jnp.zeros((n, n), dtype=A.dtype)],
        [jnp.eye(n, dtype=A.dtype), jnp.zeros((n, n), dtype=A.dtype)],
    ])
    Bz  = jnp.vstack([Sigma, jnp.zeros((n, Sigma.shape[1]), dtype=A.dtype)])
    BBt = Bz @ Bz.T
    I2n = jnp.eye(2*n, dtype=A.dtype)
    L   = jnp.kron(I2n, Az) + jnp.kron(Az, I2n)
    expLT = expm(L * T)
    rhs   = (expLT - jnp.eye(L.shape[0], dtype=A.dtype)) @ BBt.reshape(-1)
    vecC, *_ = jnp.linalg.lstsq(L, rhs, rcond=None)
    Cov = vecC.reshape(2*n, 2*n)
    CovYY = Cov[n:, n:]
    return p @ CovYY @ p

def make_discount_fn_ou(A, X0, d, p, Sigma=None, include_var=True):
    if include_var and (Sigma is not None):
        def D(t):
            logP = log_discount_det(t, A, X0, d, p)
            V    = var_integrated_z_exact(t, A, Sigma, p)
            return safe_exp(logP + 0.5 * V)
    else:
        def D(t):
            return safe_exp(log_discount_det(t, A, X0, d, p))
    return D

# -------------------------------
# Example setup
# -------------------------------
A     = jnp.array([[-0.50, 0.00, 0.00],
                   [ 0.00,-2.00, 0.00],
                   [ 0.30, 0.30,-0.80]], dtype=DTYPE)
X0    = jnp.array([0.02, 0.00, 0.03], dtype=DTYPE)
d     = jnp.array([0.0008, 0.0, 0.0], dtype=DTYPE)
p     = jnp.array([0.0, 0.0, 1.0], dtype=DTYPE)

# ✅ Fix: build Sigma as eye @ vector instead of jnp.diag
sigma_vals = jnp.array([0.008, 0.003, 0.0], dtype=DTYPE)
Sigma      = jnp.eye(3, dtype=DTYPE) * sigma_vals  # diagonal matrix

D = make_discount_fn_ou(A, X0, d, p, Sigma, include_var=True)

print("Discount(10y) =", float(D(10.0)))
print("Discount(9.9y) =", float(D(9.9)))
