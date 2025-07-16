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

def make_sliding_regression_with_penalty_fn(
    t1, t2, epsilon=1e-3, big_penalty=1e6,
    group_sparsity=False, group_by_country=False,
    n_countries=None, m_tenors=None,
    group_trigger_mode="mean", forced_group_mask=None
):
    @jax.jit
    def _sliding(X, Y):
        n_samples, d = X.shape
        m = Y.shape[1]
        n_windows = (n_samples - t1) // t2 + 1
        starts = jnp.arange(n_windows) * t2

        def extract_window(data, start):
            return jax.lax.dynamic_slice(data, (start, 0), (t1, data.shape[1]))

        X_wins = jax.vmap(lambda s: extract_window(X, s))(starts)
        Y_wins = jax.vmap(lambda s: extract_window(Y, s))(starts)

        # === First OLS pass ===
        W_ols = jax.vmap(ols_kernel)(X_wins, Y_wins)

        # === Compute penalty mask ===
        if jnp.ndim(epsilon) == 0:
            threshold = jnp.abs(W_ols) < epsilon
        else:
            threshold = jnp.abs(W_ols) < epsilon[None, None, :]  # broadcast per target

        if group_by_country:
            assert n_countries * m_tenors == d, "n_countries * m_tenors must equal num features"
            W_reshaped = W_ols.reshape((n_windows, n_countries, m_tenors, m))
            abs_W = jnp.abs(W_reshaped)  # (n_windows, n_countries, m_tenors, m)

            if group_trigger_mode == "mean":
                group_stat = jnp.mean(abs_W, axis=2)  # (n_windows, n_countries, m)
            elif group_trigger_mode == "median":
                group_stat = jnp.median(abs_W, axis=2)
            elif group_trigger_mode == "forced":
                if forced_group_mask is None:
                    raise ValueError("forced_group_mask must be provided when using group_trigger_mode='forced'")
                group_mask = jnp.broadcast_to(forced_group_mask[None, :, :], (n_windows, n_countries, m))
            else:
                raise ValueError("group_trigger_mode must be 'mean', 'median', or 'forced'")

            if group_trigger_mode != "forced":
                group_mask = group_stat < epsilon if jnp.ndim(epsilon) == 0 else group_stat < epsilon[None, None, :]

            group_mask_broadcast = jnp.repeat(group_mask, m_tenors, axis=2)  # (n_windows, n_countries, m_tenors, m)
            penalty_mask = jnp.where(group_mask_broadcast, big_penalty, 0.0).reshape((n_windows, d, m))

        elif group_sparsity:
            group_mask = jnp.mean(threshold, axis=2) > 0.5  # (n_windows, d)
            penalty_mask = jnp.where(group_mask[..., None], big_penalty, 0.0)
        else:
            penalty_mask = jnp.where(threshold, big_penalty, 0.0)

        # === Second penalized OLS pass ===
        def penalized_ols(X_win, Y_win, penalty_vec):
            XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
            XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)

            penalty_diag = jnp.mean(penalty_vec, axis=1)  # (d,)
            XtX = XtX + jnp.diag(penalty_diag)

            return solve(XtX, XtY, sym_pos=True)

        W_penalized = jax.vmap(penalized_ols)(X_wins, Y_wins, penalty_mask)

        return W_ols, W_penalized

    return _sliding


# Example: force to zero country 0,1 for all targets, leave rest
forced_group_mask = jnp.array([
    [1, 1, 1],  # country 1, all targets → penalize
    [1, 1, 1],  # country 2, all targets → penalize
    [0, 0, 0],  # country 3 → skip
    [0, 0, 0],  # country 4 → skip
    [0, 0, 0],  # country 5 → skip
])

regression_fn = make_sliding_regression_with_penalty_fn(
    t1=200, t2=50,
    epsilon=1e-3,
    group_by_country=True,
    n_countries=5, m_tenors=10,
    group_trigger_mode="forced",
    forced_group_mask=forced_group_mask
)

W_ols, W_penalized = regression_fn(X, Y)
