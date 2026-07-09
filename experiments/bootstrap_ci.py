"""Paired test-path bootstrap CI on the tail win (audit F4).

The multi-seed '+/-' is training-seed spread on one fixed test draw; it does NOT capture test-path
sampling noise. Here we hold the trained policies fixed and bootstrap the 200k TEST paths (resample
with replacement), recomputing CVaR99 and entropic risk each time, to put a proper CI on the
deep-vs-band tail gap. GBM, c=1%, entropic objective.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, Hedger, rollout_pnl, band_hedge_pnl,
    ww_band_hedge_pnl, train_hedger, entropic_risk,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S0, K, SIGMA, T, N, C, LAM = 100.0, 100.0, 0.20, 1.0, 30, 0.01, 1.0
prem = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
gv = torch.Generator(device=DEV).manual_seed(999); Sv = simulate_gbm(S0,0.,SIGMA,T,N,100_000,DEV,gv)
gt = torch.Generator(device=DEV).manual_seed(12345); St = simulate_gbm(S0,0.,SIGMA,T,N,200_000,DEV,gt)
H = torch.linspace(0,0.4,41); WW = torch.linspace(0,6,61)

def erisk_np(W): return (np.log(np.mean(np.exp(-LAM*W))))/LAM
def cvar99_np(W):
    L=-W; q=np.quantile(L,0.99); return q + 100*np.clip(L-q,0,None).mean()

# train deep (seed 0, entropic) and get per-path W
torch.manual_seed(0); hed = Hedger(n_features=5, hidden=(64,64), use_prev=True).to(DEV)
train_hedger(hed, K=K, sigma=SIGMA, T=T, N=N, cost=C, premium=prem, S0=S0,
             objective="entropic", lam=LAM, use_prev=True, steps=1600, batch=16384, lr=1e-3, device=DEV, seed=0)
hed.eval()
def erisk_t(W): return float(entropic_risk(W.double().cpu(), LAM))
bh = min(H.tolist(), key=lambda h: erisk_t(band_hedge_pnl(Sv,K,SIGMA,T,C,prem,h)))
ws = min(WW.tolist(), key=lambda s: erisk_t(ww_band_hedge_pnl(Sv,K,SIGMA,T,C,prem,s)))
with torch.no_grad():
    Wd = rollout_pnl(hed, St, K, SIGMA, T, C, prem, True).double().cpu().numpy()
Wb = band_hedge_pnl(St,K,SIGMA,T,C,prem,bh).double().cpu().numpy()
Ww = ww_band_hedge_pnl(St,K,SIGMA,T,C,prem,ws).double().cpu().numpy()

print(f"point estimates (GBM c={C}, entropic lam={LAM}):")
for nm,W in [("deep",Wd),("const band",Wb),("WW band",Ww)]:
    print(f"  {nm:<12} entropic {erisk_np(W):7.3f}  CVaR99 {cvar99_np(W):7.3f}")

# paired bootstrap over the 200k test paths
n = len(Wd); B = 1000
rng = np.random.default_rng(0)
d_cv_const, d_cv_ww, d_en_ww = [], [], []
for _ in range(B):
    idx = rng.integers(0, n, n)
    wd, wb, ww_ = Wd[idx], Wb[idx], Ww[idx]
    d_cv_const.append(cvar99_np(wb)-cvar99_np(wd))
    d_cv_ww.append(cvar99_np(ww_)-cvar99_np(wd))
    d_en_ww.append(erisk_np(ww_)-erisk_np(wd))
def ci(a): a=np.array(a); return a.mean(), np.percentile(a,2.5), np.percentile(a,97.5)
print(f"\npaired bootstrap ({B} resamples of 200k test paths), band - deep (>0 = deep better):")
for nm,a in [("CVaR99  deep vs const band", d_cv_const),
             ("CVaR99  deep vs WW band", d_cv_ww),
             ("entropic deep vs WW band", d_en_ww)]:
    m,lo,hi = ci(a); print(f"  {nm:<28} {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                            f"{'SIGNIFICANT' if lo>0 else 'n.s.'}")
