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



# Run
W_ols, W_penalized = regression_fn(X, Y)

import jax
import jax.numpy as jnp
from functools import partial

# --- Import your function here ---
# from your_module import make_sliding_regression_with_penalty_fn

# --- Parameters ---
n_samples = 1000
n_features = 7
n_countries = 5
n_tenors = 10
n_outputs = n_countries * n_tenors

t1 = 200
t2 = 50
epsilon = 0.01
big_penalty = 1e6

# --- Generate synthetic data ---
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
X = jax.random.normal(key1, (n_samples, n_features))
Y = jax.random.normal(key2, (n_samples, n_outputs)) * 0.1  # small random coefficients

# Add a few strong coefficients for testing
Y = Y.at[:, :5].set(Y[:, :5] + X[:, [0]] * 5)  # strong influence on first 5 outputs

# --- Create regression function ---
regression_fn = make_sliding_regression_with_penalty_fn(
    t1=t1,
    t2=t2,
    epsilon=epsilon,
    big_penalty=big_penalty,
    group_by_country=True,
    n_countries=n_countries,
    n_tenors=n_tenors,
    group_trigger_mode="mean"
)

# --- Run regression ---
W_ols, W_penalized = regression_fn(X, Y)  # shapes: (n_windows, 7, n_outputs)

# --- Analyze results ---
n_windows = W_ols.shape[0]

# Reshape weights for reporting: (n_windows, 7, n_countries, n_tenors)
W_ols_reshaped = W_ols.reshape((n_windows, n_features, n_countries, n_tenors))
W_pen_reshaped = W_penalized.reshape((n_windows, n_features, n_countries, n_tenors))

# Compute difference
diff = jnp.abs(W_pen_reshaped - W_ols_reshaped)

# Report example window (first one)
window_idx = 0
print(f"\n=== Window {window_idx} ===")
for i in range(n_features):
    for c in range(n_countries):
        ols_vals = W_ols_reshaped[window_idx, i, c, :]
        pen_vals = W_pen_reshaped[window_idx, i, c, :]
        changed = jnp.any(jnp.abs(ols_vals - pen_vals) > 1e-6)
        if changed:
            print(f"Feature {i}, Country {c}: PENALIZED → changed coefficients")
        else:
            print(f"Feature {i}, Country {c}: kept (no penalty)")

# Optional: show raw weights
print("\nExample raw weights (first feature, first country):")
print("OLS:", W_ols_reshaped[window_idx, 0, 0, :])
print("PEN:", W_pen_reshaped[window_idx, 0, 0, :])
