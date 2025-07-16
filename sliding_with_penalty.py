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
