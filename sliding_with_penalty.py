import jax
import jax.numpy as jnp

@jax.jit
def ols_kernel(X_win, Y_win):
    XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
    XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)
    return jnp.linalg.solve(XtX, XtY)

def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_by_country=False, n_countries=None, n_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None, top_n_per_country=None,
    freeze_non_masked=False
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

            if group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using 'forced'")
                if forced_group_mask.shape != (n_countries, n_tenors, 7):
                    raise ValueError(f"forced_group_mask must have shape ({n_countries}, {n_tenors}, 7)")
                group_mask_broadcast = jnp.broadcast_to(
                    forced_group_mask[None, :, :, :], (n_windows, n_countries, n_tenors, 7)
                ).transpose(0, 3, 1, 2)  # (n_windows, 7, n_countries, n_tenors)

            else:
                raise ValueError("This setup only covers 'forced' mode for simplicity.")

            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, 7, n_countries * n_tenors))

        else:
            threshold = jnp.abs(W_ols) < epsilon
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        def penalized_ols(X_win, Y_win, penalty_mat):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)

            def solve_per_output(XtY_col, penalties_col):
                XtX_penalized = XtX + jnp.diag(penalties_col)
                return jnp.linalg.solve(XtX_penalized, XtY_col)

            W_cols = jax.vmap(solve_per_output, in_axes=(1, 1))(XtY, penalty_mat)
            return W_cols.T  # (7, n_outputs)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        W_final = jnp.where(penalty_mask > 0, W_penalized, W_ols) if freeze_non_masked else W_penalized

        # === Compute R² helper ===
        def compute_r2(X_w, Y_w, W):
            Y_pred = jnp.einsum('wij,wjk->wik', X_w, W)  # (n_windows, t1, n_outputs)
            resid_sq = jnp.sum((Y_w - Y_pred) ** 2, axis=1)
            y_mean = jnp.mean(Y_w, axis=1, keepdims=True)
            total_sq = jnp.sum((Y_w - y_mean) ** 2, axis=1)
            return 1.0 - resid_sq / total_sq  # (n_windows, n_outputs)

        r2_ols = compute_r2(X_wins, Y_wins, W_ols)
        r2_final = compute_r2(X_wins, Y_wins, W_final)

        # Reduce over windows
        r2_ols_mean = jnp.mean(r2_ols, axis=0)      # (n_outputs,)
        r2_final_mean = jnp.mean(r2_final, axis=0)  # (n_outputs,)

        return W_ols, W_final, r2_ols_mean, r2_final_mean

    return _sliding

# Labels
country_labels = ['US', 'DE', 'FR', 'UK', 'JP']
tenor_labels = ['1Y', '2Y', '5Y', '10Y', '30Y', '1M', '3M', '6M', '12M', '20Y']

n_countries = len(country_labels)
n_tenors = len(tenor_labels)
n_features = 7

# Initialize mask
forced_group_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)

# Mask hedge 0 for US-1Y
us_idx = country_labels.index('US')
oneY_idx = tenor_labels.index('1Y')
forced_group_mask = forced_group_mask.at[us_idx, oneY_idx, 0].set(True)

# Mask hedge 3 + 4 for JP-10Y
jp_idx = country_labels.index('JP')
tenY_idx = tenor_labels.index('10Y')
forced_group_mask = forced_group_mask.at[jp_idx, tenY_idx, 3].set(True)
forced_group_mask = forced_group_mask.at[jp_idx, tenY_idx, 4].set(True)

