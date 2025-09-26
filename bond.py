import jax
import jax.numpy as jnp

def schedule_with_stub(T, freq=2, dtype=jnp.float64):
    """
    Generate coupon dates and accrual fractions up to maturity T, 
    with final stub if T isn't on a coupon date.
    Works safely with JAX (no concretization errors).
    """
    f = jnp.array(freq, dtype=dtype)
    # number of *full* coupons before maturity (floored, but as int32 array)
    n_full = jnp.floor(T * f).astype(jnp.int32)

    # full coupon payment times and accruals
    idx = jnp.arange(1, n_full + 1, dtype=jnp.int32)
    t_full = idx.astype(dtype) / f
    a_full = jnp.full_like(t_full, 1.0 / f, dtype=dtype)

    # stub accrual fraction
    last_full = n_full.astype(dtype) / f
    d_stub = T - last_full
    has_stub = d_stub > jnp.array(1e-14, dtype=dtype)

    def _with_stub(args):
        t, a, d_stub = args
        return (jnp.concatenate([t, T[None]]),
                jnp.concatenate([a, d_stub[None]]))

    def _no_stub(args):
        t, a, _ = args
        return t, a

    return jax.lax.cond(has_stub,
                        _with_stub,
                        _no_stub,
                        operand=(t_full, a_full, d_stub))

def bond_price_continuous(T, c, discount_fn, freq=2, dtype=jnp.float64):
    """
    Coupon bond price with coupon rate c, maturity T, coupon frequency freq.
    Uses final stub if T is off-schedule.
    discount_fn(t) must return discount factor D(t).
    """
    times, accruals = schedule_with_stub(T, freq=freq, dtype=dtype)
    D_times = jax.vmap(discount_fn)(times)
    coupons = jnp.sum(c * accruals * D_times)
    principal = discount_fn(T)
    return coupons + principal
