"""Classical sparse reconstruction using the same separable sensing operator."""

import numpy as np
from scipy.fft import dctn, idctn


def measure(x, h, w):
    return h @ x @ w.T


def adjoint(y, h, w):
    return h.T @ y @ w


def fista_dct(y, h, w, lam=0.005, iterations=200):
    """Minimize .5 ||H X W.T - Y||_F^2 + lam ||DCT(X)||_1.

    The orthonormal DCT makes coefficient shrinkage the exact proximal step.
    Clipping occurs only during evaluation, not inside the optimizer.
    """
    lipschitz = float(np.linalg.norm(h, 2) ** 2 * np.linalg.norm(w, 2) ** 2)
    if lipschitz <= 0 or lam < 0 or iterations < 1:
        raise ValueError('Invalid operator, regularization, or iteration budget')
    x = adjoint(y, h, w)
    z = x.copy()
    t = 1.0
    for _ in range(iterations):
        gradient = adjoint(measure(z, h, w) - y, h, w)
        coeff = dctn(z - gradient / lipschitz, norm='ortho')
        coeff = np.sign(coeff) * np.maximum(np.abs(coeff) - lam / lipschitz, 0)
        new_x = idctn(coeff, norm='ortho')
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new_x + (t - 1) / new_t * (new_x - x)
        x, t = new_x, new_t
    return x
