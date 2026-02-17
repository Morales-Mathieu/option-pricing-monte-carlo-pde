import numpy as np
from scipy.stats import norm


def _check_inputs(S, K, T, r, sigma):
    if np.any(np.asarray(S) <= 0):
        raise ValueError("S must be > 0")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("K must be > 0")
    if np.any(np.asarray(T) <= 0):
        raise ValueError("T must be > 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be > 0")


def d1_d2(S, K, T, r, sigma):
    """
    Compute d1 and d2 in the Black–Scholes model.
    Supports scalars or numpy arrays (broadcasting works).
    """
    _check_inputs(S, K, T, r, sigma)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call"):
    """
    Black–Scholes price for a European call or put.
    option: "call" or "put"
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)

    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option="call"):
    """
    Return main Black–Scholes Greeks for a European option:
    delta, gamma, vega, theta.

    Theta is returned with the convention dPrice/dT (not per day).
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)

    pdf_d1 = norm.pdf(d1)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    theta_call = -(S * pdf_d1 * sigma) / (2.0 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_put = -(S * pdf_d1 * sigma) / (2.0 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    if option == "call":
        delta = delta_call
        theta = theta_call
    elif option == "put":
        delta = delta_put
        theta = theta_put
    else:
        raise ValueError("option must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def implied_vol_bisection(S, K, T, r, market_price,
                          option="call",
                          sigma_min=1e-6,
                          sigma_max=5.0,
                          tol=1e-8,
                          max_iter=100):
    """
    Compute implied volatility using the bisection method.
    """

    def f(sigma):
        return bs_price(S, K, T, r, sigma, option=option) - market_price

    a = sigma_min
    b = sigma_max

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Bisection method fails: root not bracketed.")

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = f(mid)

        if abs(fmid) < tol:
            return mid

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    return 0.5 * (a + b)


def implied_vol_newton(S, K, T, r, market_price,
                       option="call",
                       sigma_init=0.2,
                       tol=1e-8,
                       max_iter=100):
    """
    Compute implied volatility using Newton's method.
    """

    sigma = sigma_init

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option=option)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_greeks(S, K, T, r, sigma, option=option)["vega"]

        if abs(vega) < 1e-8:
            raise ValueError("Newton method fails: Vega too small.")

        sigma = sigma - diff / vega

    return sigma


def implied_vol_bisection_path(S, K, T, r, market_price,
                               option="call",
                               sigma_min=1e-6,
                               sigma_max=5.0,
                               max_iter=60):
    """
    Same as implied_vol_bisection, but returns the sequence of (sigma, |pricing error|)
    to visualize convergence.
    """

    def f(sigma):
        return bs_price(S, K, T, r, sigma, option=option) - market_price

    a, b = sigma_min, sigma_max
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError("Bisection method fails: root not bracketed.")

    sigmas = []
    errors = []

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = f(mid)

        sigmas.append(mid)
        errors.append(abs(fmid))

        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

    return np.array(sigmas), np.array(errors)


def implied_vol_newton_path(S, K, T, r, market_price,
                            option="call",
                            sigma_init=0.2,
                            max_iter=30):
    """
    Same as implied_vol_newton, but returns the sequence of (sigma, |pricing error|)
    to visualize convergence.
    """

    sigma = float(sigma_init)

    sigmas = []
    errors = []

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option=option)
        diff = price - market_price

        sigmas.append(sigma)
        errors.append(abs(diff))

        vega = bs_greeks(S, K, T, r, sigma, option=option)["vega"]
        if abs(vega) < 1e-10:
            break

        sigma = sigma - diff / vega

    return np.array(sigmas), np.array(errors)

