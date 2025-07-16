def make_sliding_regression_with_penalty_fn(t1, t2, epsilon=1e-3, big_penalty=1e6):
    @jax.jit
    def _sliding(X, Y):
        n_samples = X.shape[0]
        n_windows = (n_samples - t1) // t2 + 1
        starts = jnp.arange(n_windows) * t2

        def extract_window(data, start):
            return jax.lax.dynamic_slice(data, (start, 0), (t1, data.shape[1]))

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)

        # === First OLS pass ===
        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)

        # === Build penalty mask ===
        penalty_mask = jnp.where(jnp.abs(W_ols) < epsilon, big_penalty, 0.0)  # (n_windows, d, m)

        # === Second penalized OLS pass ===
        def penalized_ols(X_win, Y_win, penalty_vec):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)

            # Apply penalty only on diagonal, averaged across targets
            penalty_diag = jnp.mean(penalty_vec, axis=1)  # (d,)
            XtX = XtX + jnp.diag(penalty_diag)

            return solve(XtX, XtY, sym_pos=True)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_penalized

    return _sliding



/====

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve

# --- Core OLS kernel ---
@jax.jit
def ols_kernel(X_win, Y_win):
    XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
    XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)
    return solve(XtX, XtY, sym_pos=True)

# --- Main sliding regression function ---
def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_by_country=False, n_countries=None, n_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None
):
    @jax.jit
    def _sliding(X, Y):
        n_samples, d_features = X.shape
        n_outputs = Y.shape[1]
        assert d_features == 7, "X must have 7 features"
        assert n_countries * n_tenors == n_outputs, "n_countries * n_tenors must equal Y.shape[1]"

        n_windows = (n_samples - t1) // t2 + 1
        starts = jnp.arange(n_windows) * t2

        def extract_window(data, start):
            return jax.lax.dynamic_slice(data, (start, 0), (t1, data.shape[1]))

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)  # (n_windows, t1, 7)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)  # (n_windows, t1, n_outputs)

        # === First OLS pass ===
        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)  # (n_windows, 7, n_outputs)

        # === Build penalty mask ===
        if group_by_country:
            # Reshape to (n_windows, 7, n_countries, n_tenors)
            W_reshaped = W_ols.reshape((n_windows, 7, n_countries, n_tenors))
            abs_W = jnp.abs(W_reshaped)

            if group_trigger_mode == "mean":
                group_stat = jnp.mean(abs_W, axis=3)  # (n_windows, 7, n_countries)
            elif group_trigger_mode == "median":
                group_stat = jnp.median(abs_W, axis=3)  # (n_windows, 7, n_countries)
            elif group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using 'forced'")
                # forced_group_mask shape: (7, n_countries)
                group_mask = jnp.broadcast_to(forced_group_mask[None, :, :], (n_windows, 7, n_countries))
            else:
                raise ValueError("group_trigger_mode must be 'mean', 'median', or 'forced'")

            if group_trigger_mode != "forced":
                if jnp.ndim(epsilon) == 0:
                    group_mask = group_stat < epsilon
                else:
                    group_mask = group_stat < epsilon[None, None, :]

            # Broadcast over tenors and flatten back
            group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)  # (n_windows,7,n_countries,n_tenors)
            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, 7, n_countries * n_tenors))

        else:
            # No grouping, per-coefficient thresholding
            if jnp.ndim(epsilon) == 0:
                threshold = jnp.abs(W_ols) < epsilon
            else:
                threshold = jnp.abs(W_ols) < epsilon[None, None, :]
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        # === Second penalized OLS pass ===
        def penalized_ols(X_win, Y_win, penalty_vec):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)

            penalty_diag = jnp.mean(penalty_vec, axis=1)  # (7,)
            XtX = XtX + jnp.diag(penalty_diag)

            return solve(XtX, XtY, sym_pos=True)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_ols, W_penalized

    return _sliding

regression_fn = make_sliding_regression_with_penalty_fn(
    t1=200,
    t2=50,
    epsilon=1e-3,
    big_penalty=1e6,
    group_by_country=True,
    n_countries=5,
    n_tenors=10,
    group_trigger_mode="mean",  # or "median", or "forced"
    forced_group_mask=None      # if using "forced": shape (7, n_countries)
)

# Run
W_ols, W_penalized = regression_fn(X, Y)
