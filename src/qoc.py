# quasi_optimality.py

import numpy as np
import matplotlib.pyplot as plt


def qoc(U, s, B, npoints=200, plot=True, local_min=True):
    """
    Quasi-optimality criterion for selecting the Tikhonov parameter.

    Solves:
        min_v ||G v - b||^2 + lambda * v^T G v

    with G = G1 kron I_d and G1 = U diag(s) U^T.

    Parameters
    ----------
    U : ndarray, shape (n, n)
        Eigenvectors of G1.
    s : ndarray, shape (n,)
        Eigenvalues of G1.
    B : ndarray, shape (d, n)
        Data matrix, e.g. B = (y_1 - x0, ..., y_n - x0).
    npoints : int
        Number of candidate regularization parameters.
    plot : bool
        Whether to plot the QOC curve.
    local_min : bool
        If True, choose the first interior local minimizer if available.
        Otherwise choose the global minimizer.

    Returns
    -------
    reg_c : float
        sqrt(lambda_qoc), matching the convention of your lcurve code.
    qvals : ndarray
        Quasi-optimality values.
    reg_param : ndarray
        Candidate sqrt(lambda) values.
    """

    s = np.asarray(s).reshape(-1)
    eps = np.finfo(float).eps

    # Remove tiny negative eigenvalues caused by roundoff
    s = np.maximum(s, 0.0)

    # Spectral range for lambda
    s_pos = s[s > 1000 * eps * max(1.0, np.max(s))]
    if len(s_pos) == 0:
        raise ValueError("All eigenvalues are numerically zero.")

    lam_max = np.max(s_pos)
    lam_min = max(np.min(s_pos), lam_max * 1000 * eps)

    # Candidate alpha = sqrt(lambda), from large to small
    alpha_max = np.sqrt(lam_max)
    alpha_min = np.sqrt(lam_min)
    reg_param = np.geomspace(alpha_max, alpha_min, npoints)

    # Work in the eigenbasis
    # Beta = B U, shape d x n
    Beta = B @ U

    # Compute V_lambda = B U diag(1/(s+lambda)) U^T
    # For QOC, the RKHS norm can be computed in spectral coordinates:
    #
    # ||phi_lam2 - phi_lam1||_H^2
    # = sum_j s_j * ||Beta_j * (1/(s_j+lam2)-1/(s_j+lam1))||_2^2
    #
    qvals = np.zeros(npoints - 1)

    for k in range(npoints - 1):
        lam1 = reg_param[k] ** 2
        lam2 = reg_param[k + 1] ** 2

        diff_filter = 1.0 / (s + lam2) - 1.0 / (s + lam1)

        # columns of Beta correspond to eigenvectors
        diff_coeff = Beta * diff_filter.reshape(1, -1)

        qvals[k] = np.sqrt(np.sum(s.reshape(1, -1) * diff_coeff**2))

    # Select parameter
    if local_min:
        local_indices = []
        for k in range(1, len(qvals) - 1):
            if qvals[k] <= qvals[k - 1] and qvals[k] <= qvals[k + 1]:
                local_indices.append(k)

        if len(local_indices) > 0:
            # Usually choose the first local minimum when moving from large to small lambda
            idx = local_indices[0]
        else:
            idx = int(np.argmin(qvals))
    else:
        idx = int(np.argmin(qvals))

    # qvals[k] compares reg_param[k] and reg_param[k+1].
    # A natural representative choice is the geometric midpoint.
    reg_c = np.sqrt(reg_param[idx] * reg_param[idx + 1])

    if plot:
        plt.figure()
        plt.loglog(reg_param[:-1] ** 2, qvals, "-")
        plt.axvline(reg_c**2, color="r", linestyle=":")
        plt.scatter([reg_c**2], [qvals[idx]], color="r", marker="x")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\|\phi_{\lambda_{k+1}}-\phi_{\lambda_k}\|_{H_K}$")
        plt.title(rf"Quasi-optimality criterion: $\lambda = {reg_c**2}$")

    return reg_c, qvals, reg_param