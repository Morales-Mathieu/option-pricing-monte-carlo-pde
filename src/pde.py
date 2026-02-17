import numpy as np


def setup_grid(K, T, r, sigma, S_max_mult=3.0, N=200, M=200, option="call"):
    """
    Build space/time grids and initialize terminal + boundary conditions.

    Domain: S in [0, S_max], t in [0, T]
    We store V as V[i, k] = V(S_i, t_k).

    Returns:
        S_grid, t_grid, V, dS, dt, S_max
    """
    S_max = S_max_mult * K

    dS = S_max / N
    dt = T / M

    S_grid = np.linspace(0.0, S_max, N + 1)
    t_grid = np.linspace(0.0, T, M + 1)

    V = np.zeros((N + 1, M + 1), dtype=float)

    # Terminal condition at maturity t = T (last column)
    if option == "call":
        V[:, -1] = np.maximum(S_grid - K, 0.0)
    elif option == "put":
        V[:, -1] = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")

    # Boundary conditions for all times
    # (we set them on the whole time grid so schemes can just use V[0,k], V[N,k])
    if option == "call":
        V[0, :] = 0.0
        V[-1, :] = S_max - K * np.exp(-r * (T - t_grid))
    else:
        V[0, :] = K * np.exp(-r * (T - t_grid))
        V[-1, :] = 0.0

    return S_grid, t_grid, V, dS, dt, S_max


def solve_bs_explicit(K, T, r, sigma, S_max_mult=3.0, N=200, M=200, option="call"):
    """
    Explicit finite difference scheme for the Black–Scholes PDE (backward in time).

    Returns:
        S_grid, t_grid, V
    """
    S_grid, t_grid, V, dS, dt, S_max = setup_grid(
        K, T, r, sigma, S_max_mult=S_max_mult, N=N, M=M, option=option
    )

    # interior indices (exclude boundaries 0 and N)
    i = np.arange(1, N)

    # Coefficients using S_i = i*dS (classic simplified form)
    a = 0.5 * dt * (sigma**2 * i**2 - r * i)
    b = 1.0 - dt * (sigma**2 * i**2 + r)
    c = 0.5 * dt * (sigma**2 * i**2 + r * i)

    # Backward in time: k = M-1 ... 0
    for k in range(M - 1, -1, -1):
        V[i, k] = a * V[i - 1, k + 1] + b * V[i, k + 1] + c * V[i + 1, k + 1]

    return S_grid, t_grid, V


def _solve_tridiagonal(lower, diag, upper, rhs):
    """
    Solve a tridiagonal linear system using the Thomas algorithm.

    lower: (n-1,) sub-diagonal
    diag : (n,)   main diagonal
    upper: (n-1,) super-diagonal
    rhs  : (n,)

    Returns:
        x: (n,)
    """
    n = diag.size

    # Work on copies (do not overwrite inputs)
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()
    b = diag.astype(float).copy()
    a = lower.astype(float).copy()

    # Forward sweep
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    for k in range(1, n - 1):
        denom = b[k] - a[k - 1] * c[k - 1]
        c[k] = c[k] / denom
        d[k] = (d[k] - a[k - 1] * d[k - 1]) / denom

    denom_last = b[-1] - a[-1] * c[-1]
    d[-1] = (d[-1] - a[-1] * d[-2]) / denom_last

    # Back substitution
    x = np.zeros(n, dtype=float)
    x[-1] = d[-1]
    for k in range(n - 2, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]

    return x


def solve_bs_implicit(K, T, r, sigma, S_max_mult=3.0, N=200, M=200, option="call"):
    """
    Implicit (Backward Euler) finite difference scheme for the Black–Scholes PDE.

    At each time step, we solve a tridiagonal system for the interior nodes.

    Returns:
        S_grid, t_grid, V
    """
    S_grid, t_grid, V, dS, dt, S_max = setup_grid(
        K, T, r, sigma, S_max_mult=S_max_mult, N=N, M=M, option=option
    )

    # Interior nodes i = 1..N-1 (size n = N-1)
    i = np.arange(1, N)
    n = N - 1

    # Using S_i = i*dS, standard coefficients for implicit scheme
    A = -0.5 * dt * (sigma**2 * i**2 - r * i)     # sub-diagonal (couples i-1)
    B = 1.0 + dt * (sigma**2 * i**2 + r)          # main diagonal
    C = -0.5 * dt * (sigma**2 * i**2 + r * i)     # super-diagonal (couples i+1)

    lower = A[1:]      # length n-1
    diag = B           # length n
    upper = C[:-1]     # length n-1

    # Backward in time
    for k in range(M - 1, -1, -1):
        rhs = V[i, k + 1].copy()

        # Add boundary contributions (since boundaries are known at time k)
        rhs[0]  -= A[0]  * V[0, k]
        rhs[-1] -= C[-1] * V[N, k]

        V[i, k] = _solve_tridiagonal(lower, diag, upper, rhs)

    return S_grid, t_grid, V


def solve_bs_crank_nicolson(K, T, r, sigma, S_max_mult=3.0, N=200, M=200, option="call"):
    """
    Crank–Nicolson finite difference scheme for the Black–Scholes PDE.

    Returns:
        S_grid, t_grid, V
    """
    S_grid, t_grid, V, dS, dt, S_max = setup_grid(
        K, T, r, sigma, S_max_mult=S_max_mult, N=N, M=M, option=option
    )

    # Interior nodes i = 1..N-1
    i = np.arange(1, N)
    n = N - 1

    # Using S_i = i*dS, standard CN coefficients
    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta  = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

    # Left matrix M (unknown at time k)
    A_M = -alpha          # sub
    B_M = 1.0 - beta      # main
    C_M = -gamma          # super

    # Right matrix N (known at time k+1)
    A_N = alpha
    B_N = 1.0 + beta
    C_N = gamma

    lower_M = A_M[1:]
    diag_M = B_M
    upper_M = C_M[:-1]

    # Backward in time: solve for V[:,k] from V[:,k+1]
    for k in range(M - 1, -1, -1):
        V_next = V[i, k + 1]

        # Build RHS: N * V^{k+1}
        rhs = (
            A_N * V[i - 1, k + 1] +
            B_N * V_next +
            C_N * V[i + 1, k + 1]
        )

        # Add boundary contributions from M * V^k
        rhs[0]  -= A_M[0]  * V[0, k]
        rhs[-1] -= C_M[-1] * V[N, k]

        V[i, k] = _solve_tridiagonal(lower_M, diag_M, upper_M, rhs)

    return S_grid, t_grid, V
