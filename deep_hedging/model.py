"""DEEP — the neural hedging policy, its state features, the differentiable rollout, and pathwise training.

Conventions: SHORT one European call (strike K, maturity T), receive premium p0, run a self-financing
stock hedge with holdings delta_i over [t_i, t_{i+1}], r=0. Terminal wealth
    W_T = p0 + sum_i delta_i (S_{i+1}-S_i) - sum_i cost_i - (S_T-K)^+ - liq_cost.
Trained by backprop through the simulated P&L -- no RL, no PPO noise floor.
"""
import math
import torch
import torch.nn as nn

from .data import simulate_gbm
from .risk import quadratic_risk, entropic_risk, cvar_loss


class Hedger(nn.Module):
    """Per-step feedforward hedge policy shared across time. Input features per step,
    output the stock holding delta_i (unconstrained; the net learns the [0,1] range)."""

    def __init__(self, n_features, hidden=(32, 32), use_prev=True):
        super().__init__()
        self.use_prev = use_prev
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


def make_features(S_i, K, tau, sigma, T, prev_delta, use_prev):
    """State features at one step, standardized to O(1). tau = time-to-maturity (years).
    Model-agnostic coordinates; the horizon-standardized moneyness z = log(S/K)/(sigma*sqrt(tau))
    gives the net the right *scale* near expiry (where delta steepens on the sqrt(tau) scale),
    which a plain MLP cannot form from log-moneyness and time separately."""
    logm_T = torch.log(S_i / K) / (sigma * math.sqrt(T))            # overall-scale moneyness
    tau_c = torch.clamp(tau, min=(T / 1e6))
    z_tau = torch.log(S_i / K) / (sigma * torch.sqrt(tau_c))        # horizon-standardized moneyness
    z_tau = torch.clamp(z_tau, -8.0, 8.0)
    tau_n = tau / T                                                 # normalized time-to-maturity
    feats = [logm_T, z_tau, tau_n, torch.sqrt(torch.clamp(tau_n, min=0.0))]
    if use_prev:
        feats.append(prev_delta)
    return torch.stack(feats, dim=-1)


def rollout_pnl(hedger, S, K, sigma, T, cost, premium, use_prev=True, return_turnover=False):
    """Differentiable terminal wealth W_T [B] for the neural hedger over paths S [B,N+1]."""
    B, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    device, dtype = S.device, S.dtype
    prev_delta = torch.zeros(B, device=device, dtype=dtype)
    W = torch.full((B,), premium, device=device, dtype=dtype)
    turn = torch.zeros(B, device=device, dtype=dtype)
    for i in range(N):
        tau = torch.full((B,), T - i * dt, device=device, dtype=dtype)
        feats = make_features(S[:, i], K, tau, sigma, T, prev_delta, use_prev)
        delta = hedger(feats.to(torch.float32)).to(dtype)
        W = W - cost * S[:, i] * torch.abs(delta - prev_delta)   # trade cost
        W = W + delta * (S[:, i + 1] - S[:, i])                  # hedge P&L
        turn = turn + torch.abs(delta - prev_delta)
        prev_delta = delta
    W = W - cost * S[:, N] * torch.abs(prev_delta)               # liquidate hedge at T
    turn = turn + torch.abs(prev_delta)
    W = W - torch.clamp(S[:, N] - K, min=0.0)                    # pay option payoff
    return (W, turn) if return_turnover else W


def train_hedger(hedger, *, K, sigma, T, N, cost, premium, S0=100.0, mu=0.0,
                 objective="quadratic", lam=1.0, alpha=0.99, use_prev=True,
                 steps=400, batch=8192, lr=1e-3, device="cuda", seed=0, log_every=50,
                 sim=None, rollout=None):
    """sim(B, gen) -> S [B,N+1] overrides the default GBM generator (e.g. Merton jumps).
    rollout(hedger, S) -> W overrides the default rollout_pnl (e.g. a path-dependent payoff)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(hedger.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hedger.train()
    hist = []
    for t in range(steps):
        S = simulate_gbm(S0, mu, sigma, T, N, batch, device, gen) if sim is None else sim(batch, gen)
        W = rollout_pnl(hedger, S, K, sigma, T, cost, premium, use_prev) if rollout is None else rollout(hedger, S)
        if objective == "quadratic":
            loss = quadratic_risk(W)
        elif objective == "entropic":
            loss = entropic_risk(W, lam)
        elif objective == "cvar":
            loss = cvar_loss(W, alpha)
        else:
            raise ValueError(objective)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if t % log_every == 0 or t == steps - 1:
            hist.append((t, float(loss.detach())))
    return hist
