"""RDP accountant for the Poisson-sampled Gaussian mechanism (add/remove adjacency).
Integer-order closed form of Mironov, Talwar & Zhang (2019); conversion to (eps, delta) with the
tighter bound of Balle et al. (2020), the same conversion used by Google's dp_accounting."""
import numpy as np
from scipy.special import gammaln, logsumexp
ORDERS = np.arange(2, 257)

def _log_binom(n, k): return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def rdp_sampled_gaussian(q, sigma, alpha):
    """RDP of order integer alpha for one step of the Poisson-sampled Gaussian mechanism."""
    if q >= 1.0: return alpha / (2 * sigma ** 2)
    k = np.arange(0, alpha + 1)
    logt = _log_binom(alpha, k) + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(logt) / (alpha - 1))

def eps_from_rdp(rdp, orders, delta):
    rdp = np.asarray(rdp); orders = np.asarray(orders, dtype=float)
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    return float(np.min(np.maximum(eps, 0.0)))

def epsilon(q, sigma, steps, delta=1e-5):
    rdp = np.array([steps * rdp_sampled_gaussian(q, sigma, a) for a in ORDERS])
    return eps_from_rdp(rdp, ORDERS, delta)

def sigma_for(target_eps, q_list, steps_list, delta=1e-5):
    """Smallest sigma such that EVERY client's epsilon is <= target (worst client governs)."""
    f = lambda s: max(epsilon(q, s, st, delta) for q, st in zip(q_list, steps_list))
    lo, hi = 0.2, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(mid) > target_eps: lo = mid
        else: hi = mid
    return hi
