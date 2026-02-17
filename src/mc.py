import numpy as np


def simulate_gbm_terminal(S0, T, r, sigma, n_paths, rng=None):
    """
    Simulate terminal values S_T under the risk-neutral GBM model.

    S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def payoff_european(S, K, option="call"):
    """
    Vectorized payoff for European call/put given terminal prices S.
    """
    if K <= 0:
        raise ValueError("K must be > 0")

    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)

    raise ValueError("option must be 'call' or 'put'")


def mc_price_plain(S0, K, T, r, sigma, n_paths, option="call", rng=None):
    """
    Plain Monte Carlo estimator for a European option price.
    Returns (price, standard_error).
    """
    ST = simulate_gbm_terminal(S0, T, r, sigma, n_paths, rng=rng)
    payoffs = payoff_european(ST, K, option=option)

    disc = np.exp(-r * T)
    price = disc * payoffs.mean()

    # Standard error of discounted payoff mean
    se = disc * payoffs.std(ddof=1) / np.sqrt(n_paths)
    return price, se


def mc_price_antithetic(S0, K, T, r, sigma, n_paths, option="call", rng=None):
    """
    Antithetic variates Monte Carlo estimator.
    Uses pairs (Z, -Z). n_paths is rounded up to an even number.
    Returns (price, standard_error).
    """
    if n_paths % 2 == 1:
        n_paths = n_paths + 1

    if rng is None:
        rng = np.random.default_rng()

    half = n_paths // 2
    z = rng.standard_normal(half)
    z_pair = np.concatenate([z, -z])

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z_pair
    ST = S0 * np.exp(drift + diffusion)

    payoffs = payoff_european(ST, K, option=option)

    disc = np.exp(-r * T)
    price = disc * payoffs.mean()
    se = disc * payoffs.std(ddof=1) / np.sqrt(n_paths)
    return price, se


def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths, rng=None):
    """
    Simulate full GBM paths on a uniform time grid using the exact discretization.

    Returns:
        t: shape (n_steps+1,)
        S: shape (n_paths, n_steps+1)
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    Z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
    return t, S
