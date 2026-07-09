"""BASELINES — the classical hedges the deep policy is measured against (all r=0).

Black-Scholes analytics (price/delta/gamma), and the deployable classical hedges on matched paths:
naive delta, Leland modified-vol delta, the edge-reflecting no-trade band (Davis-Norman / Whalley-Wilmott
singular control), and the state-dependent Whalley-Wilmott gamma-band. Each returns terminal wealth W [B]
(short one European call). The exact intractable optimum (Hodges-Neuberger DP) is computed separately in
experiments/hn_dp.py.
"""
import math
import torch

SQRT2 = math.sqrt(2.0)


def _ncdf(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def bs_price_call(S, K, tau, sigma):
    """European call price, r=0. tau = time to maturity. Handles tau->0 cleanly."""
    S = torch.as_tensor(S, dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64)
    intrinsic = torch.clamp(S - K, min=0.0)
    safe = tau > 1e-12
    tau_s = torch.where(safe, tau, torch.ones_like(tau))
    vol = sigma * torch.sqrt(tau_s)
    d1 = (torch.log(S / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    price = S * _ncdf(d1) - K * _ncdf(d2)
    return torch.where(safe, price, intrinsic)


def bs_delta_call(S, K, tau, sigma):
    """Call delta = N(d1), r=0. tau->0 -> Heaviside(S-K)."""
    S = torch.as_tensor(S, dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64)
    safe = tau > 1e-12
    tau_s = torch.where(safe, tau, torch.ones_like(tau))
    vol = sigma * torch.sqrt(tau_s)
    d1 = (torch.log(S / K) + 0.5 * vol * vol) / vol
    delta = _ncdf(d1)
    return torch.where(safe, delta, (S > K).to(S.dtype))


def bs_gamma_call(S, K, tau, sigma):
    """BS gamma = phi(d1)/(S sigma sqrt(tau)), r=0; ->0 at tau->0."""
    S = torch.as_tensor(S, dtype=torch.float64); tau = torch.as_tensor(tau, dtype=torch.float64)
    safe = tau > 1e-9
    tau_s = torch.where(safe, tau, torch.ones_like(tau))
    vol = sigma * torch.sqrt(tau_s)
    d1 = (torch.log(S / K) + 0.5 * vol * vol) / vol
    phi = torch.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    g = phi / (S * vol)
    return torch.where(safe, g, torch.zeros_like(g))


def leland_vol(sigma, cost, dt):
    """Leland (1985) modified hedging vol under round-trip proportional cost 2*cost. Inflated vol
    flattens delta -> less rebalancing. Feed as vol_hedge to delta_hedge_pnl."""
    A = math.sqrt(2 / math.pi) * (2 * cost) / (sigma * math.sqrt(dt))
    return sigma * math.sqrt(1.0 + A)


def delta_hedge_pnl(S, K, sigma, T, cost, premium, vol_hedge=None, return_turnover=False):
    """Classical Black-Scholes delta-hedge terminal wealth on the SAME paths (baseline).
    vol_hedge lets Leland use a modified vol; defaults to true sigma."""
    vh = sigma if vol_hedge is None else vol_hedge
    B, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    device, dtype = S.device, S.dtype
    prev = torch.zeros(B, device=device, dtype=dtype)
    W = torch.full((B,), premium, device=device, dtype=dtype)
    turn = torch.zeros(B, device=device, dtype=dtype)
    for i in range(N):
        tau = T - i * dt
        delta = bs_delta_call(S[:, i], K, tau, vh)
        W = W - cost * S[:, i] * torch.abs(delta - prev)
        W = W + delta * (S[:, i + 1] - S[:, i])
        turn = turn + torch.abs(delta - prev)
        prev = delta
    W = W - cost * S[:, N] * torch.abs(prev)
    turn = turn + torch.abs(prev)
    W = W - torch.clamp(S[:, N] - K, min=0.0)
    return (W, turn) if return_turnover else W


def band_hedge_pnl(S, K, sigma, T, cost, premium, h, return_turnover=False):
    """Classical no-trade BAND around BS delta -- the Davis-Norman / Whalley-Wilmott singular-control
    optimum under proportional cost: hold while the position stays within [delta-h, delta+h], and when
    it exits, trade the MINIMUM to return to the nearest BOUNDARY (reflect to the edge), NOT snap to the
    delta center. Edge-reflection is exactly a clamp of the held position to the band. h tuned to the
    objective. (An earlier version snapped to the center, which over-trades and handicaps this baseline.)"""
    B, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    device, dtype = S.device, S.dtype
    prev = torch.zeros(B, device=device, dtype=dtype)
    W = torch.full((B,), premium, device=device, dtype=dtype)
    turn = torch.zeros(B, device=device, dtype=dtype)
    for i in range(N):
        tau = T - i * dt
        target = bs_delta_call(S[:, i], K, tau, sigma)
        delta = torch.clamp(prev, target - h, target + h)       # reflect to nearest band edge
        W = W - cost * S[:, i] * torch.abs(delta - prev)
        W = W + delta * (S[:, i + 1] - S[:, i])
        turn = turn + torch.abs(delta - prev)
        prev = delta
    W = W - cost * S[:, N] * torch.abs(prev)
    turn = turn + torch.abs(prev)
    W = W - torch.clamp(S[:, N] - K, min=0.0)
    return (W, turn) if return_turnover else W


def ww_band_hedge_pnl(S, K, sigma, T, cost, premium, scale, return_turnover=False):
    """Whalley-Wilmott (1997) state-dependent no-trade band -- the leading-order utility-indifference
    (Hodges-Neuberger) optimum under proportional cost: half-width w_i = scale * (cost * S_i * Gamma_i^2)^(1/3)
    around the BS delta, edge-reflecting. `scale` absorbs the risk-aversion constant and is tuned to the
    objective. This is the STRONG, correctly-structured classical opponent (stronger than a constant band)."""
    B, Np1 = S.shape; N = Np1 - 1; dt = T / N
    device, dtype = S.device, S.dtype
    prev = torch.zeros(B, device=device, dtype=dtype)
    W = torch.full((B,), premium, device=device, dtype=dtype)
    turn = torch.zeros(B, device=device, dtype=dtype)
    for i in range(N):
        tau = T - i * dt
        target = bs_delta_call(S[:, i], K, tau, sigma)
        gamma = bs_gamma_call(S[:, i], K, tau, sigma)
        w = scale * torch.clamp(cost * S[:, i] * gamma * gamma, min=0.0) ** (1.0 / 3.0)
        delta = torch.clamp(prev, target - w, target + w)
        W = W - cost * S[:, i] * torch.abs(delta - prev)
        W = W + delta * (S[:, i + 1] - S[:, i])
        turn = turn + torch.abs(delta - prev)
        prev = delta
    W = W - cost * S[:, N] * torch.abs(prev); turn = turn + torch.abs(prev)
    W = W - torch.clamp(S[:, N] - K, min=0.0)
    return (W, turn) if return_turnover else W
