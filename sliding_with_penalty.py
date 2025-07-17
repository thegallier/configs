import jax
import jax.numpy as jnp

@jax.jit
def ols_kernel(X_win, Y_win):
    XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
    XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)
    return jnp.linalg.solve(XtX, XtY)            # (7,n_outputs)

def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_by_country=False, n_countries=None, n_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None, top_n_per_country=None
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

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)

        # First OLS pass
        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)  # (n_windows, 7, n_outputs)

        # Build penalty mask
        if group_by_country:
            W_reshaped = W_ols.reshape((n_windows, 7, n_countries, n_tenors))
            abs_W = jnp.abs(W_reshaped)

            if group_trigger_mode == "mean":
                group_stat = jnp.mean(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "median":
                group_stat = jnp.median(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using 'forced'")
                group_mask = jnp.broadcast_to(forced_group_mask[None, :, :], (n_windows, 7, n_countries))
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "top_n":
                if top_n_per_country is None:
                    raise ValueError("top_n_per_country must be provided for 'top_n' mode")
                sorted_idx = jnp.argsort(abs_W, axis=3)
                n_drop = n_tenors - top_n_per_country
                drop_idx = sorted_idx[..., :n_drop]

                keep_mask = jnp.ones_like(abs_W, dtype=bool)

                def mark_drops(keep_c, drop_c):
                    return keep_c.at[drop_c].set(False)

                keep_mask = jax.vmap(
                    lambda keep_w, drop_w: jax.vmap(
                        lambda keep_f, drop_f: jax.vmap(
                            mark_drops, in_axes=(0, 0)
                        )(keep_f, drop_f),
                        in_axes=(0, 0)
                    )(keep_w, drop_w),
                    in_axes=(0, 0)
                )(keep_mask, drop_idx)

                group_mask_broadcast = ~keep_mask

            else:
                raise ValueError("Invalid group_trigger_mode")

            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, 7, n_countries * n_tenors))

        else:
            threshold = jnp.abs(W_ols) < epsilon
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        # Second penalized OLS pass (per-output penalties)
        def penalized_ols(X_win, Y_win, penalty_mat):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)

            def solve_per_output(XtY_col, penalties_col):
                XtX_penalized = XtX + jnp.diag(penalties_col)
                return jnp.linalg.solve(XtX_penalized, XtY_col)  # (7,)

            W_cols = jax.vmap(solve_per_output, in_axes=(1, 1))(XtY, penalty_mat)  # (n_outputs, 7)
            return W_cols.T  # (7, n_outputs)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_ols, W_penalized

    return _sliding

import jax
import jax.numpy as jnp

@jax.jit
def ols_kernel(X_win, Y_win):
    XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
    XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)
    return jnp.linalg.solve(XtX, XtY)            # (7,n_outputs)

def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_by_country=False, n_countries=None, n_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None, top_n_per_country=None
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

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)

        # First OLS pass
        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)  # (n_windows, 7, n_outputs)

        # Build penalty mask
        if group_by_country:
            W_reshaped = W_ols.reshape((n_windows, 7, n_countries, n_tenors))
            abs_W = jnp.abs(W_reshaped)

            if group_trigger_mode == "mean":
                group_stat = jnp.mean(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "median":
                group_stat = jnp.median(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using 'forced'")
                group_mask = jnp.broadcast_to(forced_group_mask[None, :, :], (n_windows, 7, n_countries))
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "top_n":
                if top_n_per_country is None:
                    raise ValueError("top_n_per_country must be provided for 'top_n' mode")
                sorted_idx = jnp.argsort(abs_W, axis=3)
                n_drop = n_tenors - top_n_per_country
                drop_idx = sorted_idx[..., :n_drop]

                keep_mask = jnp.ones_like(abs_W, dtype=bool)

                def mark_drops(keep_c, drop_c):
                    return keep_c.at[drop_c].set(False)

                keep_mask = jax.vmap(
                    lambda keep_w, drop_w: jax.vmap(
                        lambda keep_f, drop_f: jax.vmap(
                            mark_drops, in_axes=(0, 0)
                        )(keep_f, drop_f),
                        in_axes=(0, 0)
                    )(keep_w, drop_w),
                    in_axes=(0, 0)
                )(keep_mask, drop_idx)

                group_mask_broadcast = ~keep_mask

            else:
                raise ValueError("Invalid group_trigger_mode")

            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, 7, n_countries * n_tenors))

        else:
            threshold = jnp.abs(W_ols) < epsilon
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        # Second penalized OLS pass (per-output penalties)
        def penalized_ols(X_win, Y_win, penalty_mat):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)

            def solve_per_output(XtY_col, penalties_col):
                XtX_penalized = XtX + jnp.diag(penalties_col)
                return jnp.linalg.solve(XtX_penalized, XtY_col)  # (7,)

            W_cols = jax.vmap(solve_per_output, in_axes=(1, 1))(XtY, penalty_mat)  # (n_outputs, 7)
            return W_cols.T  # (7, n_outputs)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_ols, W_penalized

    return _sliding
    
# --- Parameters ---
n_samples = 1000
n_features = 7
n_countries = 5
n_tenors = 10
n_outputs = n_countries * n_tenors

t1 = 200
t2 = 50
top_n_per_country = 3
big_penalty = 1e6

# --- Generate synthetic data ---
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
X = jax.random.normal(key1, (n_samples, n_features))
Y = jax.random.normal(key2, (n_samples, n_outputs)) * 0.1

# Add strong signal for testing (feature 0 → first 5 outputs)
Y = Y.at[:, :5].set(Y[:, :5] + X[:, [0]] * 5)

# --- Get function ---
regression_fn = make_sliding_regression_with_penalty_fn(
    t1=t1,
    t2=t2,
    big_penalty=big_penalty,
    group_by_country=True,
    n_countries=n_countries,
    n_tenors=n_tenors,
    group_trigger_mode="top_n",
    top_n_per_country=top_n_per_country
)

# --- Run ---
W_ols, W_penalized = regression_fn(X, Y)

# --- Inspect ---
n_windows = W_ols.shape[0]
W_ols_reshaped = W_ols.reshape((n_windows, n_features, n_countries, n_tenors))
W_pen_reshaped = W_penalized.reshape((n_windows, n_features, n_countries, n_tenors))

window_idx = 0
print(f"\n=== Window {window_idx} Report ===")
for i in range(n_features):
    for c in range(n_countries):
        ols_vals = W_ols_reshaped[window_idx, i, c, :]
        pen_vals = W_pen_reshaped[window_idx, i, c, :]
        n_kept = jnp.sum(jnp.abs(pen_vals) > 1e-6)
        print(f"Feature {i}, Country {c}: kept {n_kept}/{n_tenors} (expected {top_n_per_country})")

/=====
import jax
import jax.numpy as jnp

@jax.jit
def ols_kernel(X_win, Y_win):
    XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
    XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)
    return jnp.linalg.solve(XtX, XtY)            # (7,n_outputs)

def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_by_country=False, n_countries=None, n_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None, top_n_per_country=None
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

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)

        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)  # (n_windows, 7, n_outputs)

        if group_by_country:
            W_reshaped = W_ols.reshape((n_windows, 7, n_countries, n_tenors))
            abs_W = jnp.abs(W_reshaped)

            if group_trigger_mode == "mean":
                group_stat = jnp.mean(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "median":
                group_stat = jnp.median(abs_W, axis=3)
                group_mask = group_stat < epsilon
                group_mask_broadcast = jnp.repeat(group_mask[..., None], n_tenors, axis=3)

            elif group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using 'forced'")
                if forced_group_mask.shape != (n_countries, n_tenors, 7):
                    raise ValueError(f"forced_group_mask must have shape ({n_countries}, {n_tenors}, 7)")
                group_mask_broadcast = jnp.broadcast_to(
                    forced_group_mask[None, :, :, :], (n_windows, n_countries, n_tenors, 7)
                ).transpose(0, 3, 1, 2)  # (n_windows, 7, n_countries, n_tenors)

            elif group_trigger_mode == "top_n":
                if top_n_per_country is None:
                    raise ValueError("top_n_per_country must be provided for 'top_n' mode")
                sorted_idx = jnp.argsort(abs_W, axis=3)
                n_drop = n_tenors - top_n_per_country
                drop_idx = sorted_idx[..., :n_drop]

                keep_mask = jnp.ones_like(abs_W, dtype=bool)

                def mark_drops(keep_c, drop_c):
                    return keep_c.at[drop_c].set(False)

                keep_mask = jax.vmap(
                    lambda keep_w, drop_w: jax.vmap(
                        lambda keep_f, drop_f: jax.vmap(
                            mark_drops, in_axes=(0, 0)
                        )(keep_f, drop_f),
                        in_axes=(0, 0)
                    )(keep_w, drop_w),
                    in_axes=(0, 0)
                )(keep_mask, drop_idx)

                group_mask_broadcast = ~keep_mask

            else:
                raise ValueError("Invalid group_trigger_mode")

            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, 7, n_countries * n_tenors))

        else:
            threshold = jnp.abs(W_ols) < epsilon
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        def penalized_ols(X_win, Y_win, penalty_mat):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)  # (7,7)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)  # (7,n_outputs)

            def solve_per_output(XtY_col, penalties_col):
                XtX_penalized = XtX + jnp.diag(penalties_col)
                return jnp.linalg.solve(XtX_penalized, XtY_col)  # (7,)

            W_cols = jax.vmap(solve_per_output, in_axes=(1, 1))(XtY, penalty_mat)  # (n_outputs, 7)
            return W_cols.T  # (7, n_outputs)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_ols, W_penalized

    return _sliding


import jax
import jax.numpy as jnp

# --- Parameters ---
n_samples = 1000
n_features = 7
n_countries = 5
n_tenors = 10
n_outputs = n_countries * n_tenors

t1 = 200
t2 = 50
big_penalty = 1e6

# --- Generate synthetic data ---
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
X = jax.random.normal(key1, (n_samples, n_features))
Y = jax.random.normal(key2, (n_samples, n_outputs)) * 0.1

# Add strong signal for testing (feature 0 → first 5 outputs)
Y = Y.at[:, :5].set(Y[:, :5] + X[:, [0]] * 5)

# --- Create forced group mask (n_countries, n_tenors, 7) ---
forced_group_mask = jnp.zeros((n_countries, n_tenors, 7), dtype=bool)
forced_group_mask = forced_group_mask.at[0, 0, 0].set(True)  # Penalize hedge 0, country 0, tenor 0
forced_group_mask = forced_group_mask.at[1, 5, 3].set(True)  # Penalize hedge 3, country 1, tenor 5

# --- Get function ---
regression_fn = make_sliding_regression_with_penalty_fn(
    t1=t1,
    t2=t2,
    big_penalty=big_penalty,
    group_by_country=True,
    n_countries=n_countries,
    n_tenors=n_tenors,
    group_trigger_mode="forced",
    forced_group_mask=forced_group_mask
)

# --- Run ---
W_ols, W_penalized = regression_fn(X, Y)

# --- Inspect ---
n_windows = W_ols.shape[0]
W_ols_reshaped = W_ols.reshape((n_windows, n_features, n_countries, n_tenors))
W_pen_reshaped = W_penalized.reshape((n_windows, n_features, n_countries, n_tenors))

window_idx = 0
print(f"\n=== Window {window_idx} Report ===")
for i in range(n_features):
    for c in range(n_countries):
        for t in range(n_tenors):
            ols_val = W_ols_reshaped[window_idx, i, c, t]
            pen_val = W_pen_reshaped[window_idx, i, c, t]
            changed = jnp.abs(ols_val - pen_val) > 1e-6
            if forced_group_mask[c, t, i]:
                status = "PENALIZED" if changed else "!!! ERROR: NOT PENALIZED"
            else:
                status = "kept" if not changed else "!!! WARNING: CHANGED"
            print(f"Feature {i}, Country {c}, Tenor {t}: {status}")

# Example raw weights
print("\nExample raw weights (feature 0, country 0):")
print("OLS:", W_ols_reshaped[window_idx, 0, 0, :])
print("PEN:", W_pen_reshaped[window_idx, 0, 0, :])
