"""Transformer deep hedging of a path-dependent (arithmetic Asian) option -- FAIR comparison.

Payoff = (mean(S_1..S_N) - K)^+  depends on the WHOLE path. Under GBM the arithmetic-Asian hedge
is Markov in (S_i, running-average, tau) -- so the informed FF handed that statistic is the
theoretical CEILING at c=0 and cannot be beaten. The scientific question is therefore NOT "can the
transformer beat the informed FF" (it can't, in principle) but: can a causal transformer, reading only
RAW price history, RECOVER that ceiling -- and does the path-blind myopic FF (spot+time only) fail?

Fairness fixes (per audit): all nets get z_tau; FF capacity matched to the transformer (~18k params);
3 seeds with spread; equal training budget.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import simulate_gbm, Hedger

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S0, K, SIGMA, T, N = 100.0, 100.0, 0.20, 1.0, 30
dt = T / N
SEEDS = [0, 1, 2]

def z_of(Si, tau_years):
    tau_c = torch.clamp(tau_years, min=T/1e6)
    return torch.clamp(torch.log(Si/K)/(SIGMA*torch.sqrt(tau_c)), -8.0, 8.0)

def tok_features(S):
    """Per-step causal token features [logm_T, z_tau, tau_n, sqrt_tau_n]. [B,N,4]."""
    B = S.shape[0]; idx = torch.arange(N, device=S.device)
    Si = S[:, :N]
    logm = torch.log(Si/K)/(SIGMA*math.sqrt(T))
    tau_y = (T - idx*dt).unsqueeze(0).expand(B, N)
    tau_n = tau_y/T
    return torch.stack([logm, z_of(Si, tau_y), tau_n, torch.sqrt(torch.clamp(tau_n,min=0.0))], dim=-1)

def asian_payoff(S): return torch.clamp(S[:, 1:].mean(dim=1) - K, min=0.0)

def rollout_ff(hedger, S, cost, premium, mode):
    B = S.shape[0]
    prev = torch.zeros(B, device=S.device, dtype=S.dtype)
    W = torch.full((B,), premium, device=S.device, dtype=S.dtype)
    locked = torch.zeros(B, device=S.device, dtype=S.dtype)
    for i in range(N):
        if i >= 1: locked = locked + S[:, i]
        tau_y = torch.full((B,), T - i*dt, device=S.device, dtype=S.dtype)
        logm = torch.log(S[:, i]/K)/(SIGMA*math.sqrt(T)); tau_n = tau_y/T
        feats = [logm, z_of(S[:, i], tau_y), tau_n, torch.sqrt(torch.clamp(tau_n,min=0.0)), prev]
        if mode == "informed": feats.append(locked/(N*K))
        d = hedger(torch.stack(feats, -1).to(torch.float32)).to(S.dtype)
        W = W - cost*S[:, i]*torch.abs(d-prev); W = W + d*(S[:, i+1]-S[:, i]); prev = d
    W = W - cost*S[:, N]*torch.abs(prev) - asian_payoff(S)
    return W

class TransformerHedger(nn.Module):
    def __init__(self, d_in=4, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, N, d_model)*0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True,
                                         activation="gelu", dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, layers); self.head = nn.Linear(d_model, 1)
        self.register_buffer("mask", torch.triu(torch.ones(N, N)*float("-inf"), diagonal=1))
    def forward(self, tok):
        h = self.tf(self.inp(tok)+self.pos, mask=self.mask)
        return self.head(h).squeeze(-1)

def rollout_transformer(hedger, S, cost, premium):
    d_all = hedger(tok_features(S).to(torch.float32)).to(S.dtype)
    B = S.shape[0]; prev = torch.zeros(B, device=S.device, dtype=S.dtype)
    W = torch.full((B,), premium, device=S.device, dtype=S.dtype)
    for i in range(N):
        d = d_all[:, i]
        W = W - cost*S[:, i]*torch.abs(d-prev); W = W + d*(S[:, i+1]-S[:, i]); prev = d
    W = W - cost*S[:, N]*torch.abs(prev) - asian_payoff(S)
    return W

def rmse(W): W = W.double().cpu(); return float(torch.sqrt((W*W).mean()))
def nparams(m): return sum(p.numel() for p in m.parameters())

def train(kind, hedger, cost, premium, steps=4000, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.Adam(hedger.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for t in range(steps):
        S = simulate_gbm(S0, 0.0, SIGMA, T, N, 8192, DEV, gen)
        W = rollout_transformer(hedger, S, cost, premium) if kind == "transformer" else rollout_ff(hedger, S, cost, premium, kind)
        loss = (W*W).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()

gp = torch.Generator(device=DEV).manual_seed(321)
premium = float(asian_payoff(simulate_gbm(S0,0.0,SIGMA,T,N,400_000,DEV,gp)).double().mean())
gt = torch.Generator(device=DEV).manual_seed(12345)
S_test = simulate_gbm(S0,0.0,SIGMA,T,N,200_000,DEV,gt)
print(f"Arithmetic Asian call K={K} sigma={SIGMA} T={T} N={N} fair premium={premium:.4f} device={DEV}  seeds={SEEDS}")

def build(kind):
    if kind == "myopic":   return Hedger(n_features=5, hidden=(128,128), use_prev=True).to(DEV)
    if kind == "informed": return Hedger(n_features=6, hidden=(128,128), use_prev=True).to(DEV)
    return TransformerHedger().to(DEV)
print("params:", {k: nparams(build(k)) for k in ("myopic","informed","transformer")})

for cost in (0.0, 0.01):
    print(f"\n===== cost c = {cost:.1%} =====")
    res = {}
    for kind in ("myopic","informed","transformer"):
        rs = []
        for s in SEEDS:
            torch.manual_seed(s); hed = build(kind); train(kind, hed, cost, premium, seed=s); hed.eval()
            with torch.no_grad():
                W = rollout_transformer(hed, S_test, cost, premium) if kind=="transformer" else rollout_ff(hed, S_test, cost, premium, kind)
            rs.append(rmse(W))
        res[kind] = np.array(rs)
        print(f"  {kind:<20} RMSE {res[kind].mean():.3f} +- {res[kind].std(ddof=1):.3f}  (seeds {np.round(res[kind],3)})")
    mvi = 100*(res['myopic'].mean()-res['transformer'].mean())/res['myopic'].mean()
    tvi = 100*(res['transformer'].mean()-res['informed'].mean())/res['informed'].mean()
    print(f"  transformer vs myopic FF (path matters): {mvi:+.1f}%   "
          f"transformer vs informed-FF ceiling: {tvi:+.1f}% (>=0 or ~0 => recovers/holds the ceiling)")
