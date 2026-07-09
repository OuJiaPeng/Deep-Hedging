"""DATA — differentiable market simulators (path generation) + real-data acquisition.

The project trains on *generated* paths (a net needs millions; reality gives one history) and tests
on real held-out paths. Both live here: GBM / Merton simulators, and a cached Yahoo fetch.
"""
import os
import math
import json
import urllib.request
import numpy as np
import torch


def simulate_gbm(S0, mu, sigma, T, N, B, device, gen=None, dtype=torch.float64):
    """Simulate B GBM paths with N steps. Returns S [B, N+1]. r=0; mu is the (physical)
    drift used only to generate paths (hedging risk is measure-agnostic here; mu=0 default)."""
    dt = T / N
    Z = torch.randn(B, N, device=device, dtype=dtype, generator=gen)
    logincr = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * Z
    logS = torch.cumsum(logincr, dim=1)
    S = torch.empty(B, N + 1, device=device, dtype=dtype)
    S[:, 0] = S0
    S[:, 1:] = S0 * torch.exp(logS)
    return S


def simulate_merton(S0, mu, sigma, T, N, B, device, gen=None,
                    lam=1.0, mj=-0.05, sj=0.10, dtype=torch.float64):
    """Merton jump-diffusion: GBM + compound-Poisson log jumps ~ N(mj, sj^2), intensity lam/yr.
    Risk-neutral compensator keeps E[S_t]=S0*exp(mu t). Incomplete market: delta cannot hedge
    the jump gap risk, so the deep hedge's edge should widen here."""
    dt = T / N
    k = math.exp(mj + 0.5 * sj * sj) - 1.0                      # E[e^J - 1]
    Z = torch.randn(B, N, device=device, dtype=dtype, generator=gen)
    counts = torch.poisson(torch.full((B, N), lam * dt, device=device, dtype=dtype), generator=gen)
    Zj = torch.randn(B, N, device=device, dtype=dtype, generator=gen)
    jump = counts * mj + torch.sqrt(torch.clamp(counts, min=0.0)) * sj * Zj
    logincr = (mu - 0.5 * sigma * sigma - lam * k) * dt + sigma * math.sqrt(dt) * Z + jump
    logS = torch.cumsum(logincr, dim=1)
    S = torch.empty(B, N + 1, device=device, dtype=dtype)
    S[:, 0] = S0
    S[:, 1:] = S0 * torch.exp(logS)
    return S


def fetch(sym, cache_dir=None):
    """Daily close prices for `sym` from Yahoo's chart endpoint (~25y). Cached to cache_dir
    (default <repo>/artifacts) as real_{sym}.npy so reruns don't refetch. Returns 1-D np.array of closes."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"real_{sym}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=25y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    close = d["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    arr = np.array([c for c in close if c is not None], dtype=np.float64)
    np.save(cache, arr)
    return arr
