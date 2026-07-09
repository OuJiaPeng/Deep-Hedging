"""Where is the genuine win? Separate TAIL-OBJECTIVE from MARKET-INCOMPLETENESS.

On RMSE (mean-variance) the deep hedge only MATCHES the correctly-implemented no-trade band --
under proportional cost the band is the optimum (Whalley-Wilmott), so that is a boundary, not a win.
A real win requires a setting where the delta-band is STRUCTURALLY unable to be optimal. Hypothesis:
that setting is a TAIL objective in an INCOMPLETE (jump) market -- a delta-tracking band cannot hedge
jump gap risk at any width, so a tail-trained deep hedge should beat the best band under jumps, while
merely tying it under pure GBM (complete market).

Train objective = entropic (exp-utility, tail-sensitive, = the entropy-regularized-control lens);
report CVaR99 (interpretable tail) + RMSE. Band tuned on the SAME entropic objective (fair). 3 seeds.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, simulate_merton, Hedger, rollout_pnl,
    delta_hedge_pnl, band_hedge_pnl, ww_band_hedge_pnl, train_hedger, entropic_risk,
)
WW_GRID = torch.linspace(0.0, 6.0, 61)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S0, K, SIGMA, T, N, C = 100.0, 100.0, 0.20, 1.0, 30, 0.01
LAM = 1.0
SEEDS = [0, 1, 2]
JP = dict(lam=1.0, mj=-0.05, sj=0.10)
H_GRID = torch.linspace(0.0, 0.40, 41)

def erisk(W): return float(entropic_risk(W.double().cpu(), LAM))
def rmse(W): W = W.double().cpu(); return float(torch.sqrt((W*W).mean()))
def cvar99(W): W=W.double().cpu(); L=-W; q=torch.quantile(L,0.99); return float(q+100*torch.clamp(L-q,min=0).mean())

MARKETS = {
    "GBM (complete)":  (lambda B, g: simulate_gbm(S0,0.0,SIGMA,T,N,B,DEV,g),  bs_price_call(torch.tensor(S0),K,T,SIGMA).item()),
    "Merton (jumps)":  (lambda B, g: simulate_merton(S0,0.0,SIGMA,T,N,B,DEV,g,**JP), None),
}

print(f"WIN TEST  entropic objective (lambda={LAM}), report CVaR99+RMSE, cost={C}, seeds={SEEDS}")
for mname, (sim, prem) in MARKETS.items():
    gv = torch.Generator(device=DEV).manual_seed(999); Sv = sim(100_000, gv)
    gt = torch.Generator(device=DEV).manual_seed(12345); St = sim(200_000, gt)
    if prem is None:  # fair premium by MC for the jump market
        prem = float(torch.clamp(St[:, -1]-K, min=0).double().mean())
    # classical: naive delta + entropic-tuned band
    Wd = delta_hedge_pnl(St, K, SIGMA, T, C, prem)
    bh, br = 0.0, 1e18
    for h in H_GRID.tolist():
        r = erisk(band_hedge_pnl(Sv, K, SIGMA, T, C, prem, h))
        if r < br: br, bh = r, h
    Wb = band_hedge_pnl(St, K, SIGMA, T, C, prem, bh)
    # STRONG classical: Whalley-Wilmott state-dependent (gamma-scaled) band, tuned on entropic
    ws, wr = 0.0, 1e18
    for sc in WW_GRID.tolist():
        r = erisk(ww_band_hedge_pnl(Sv, K, SIGMA, T, C, prem, sc))
        if r < wr: wr, ws = r, sc
    Www = ww_band_hedge_pnl(St, K, SIGMA, T, C, prem, ws)
    # deep, 3 seeds, trained on this market with entropic objective
    de, dc, dr = [], [], []
    for s in SEEDS:
        torch.manual_seed(s); hed = Hedger(n_features=5, hidden=(64,64), use_prev=True).to(DEV)
        train_hedger(hed, K=K, sigma=SIGMA, T=T, N=N, cost=C, premium=prem, S0=S0,
                     objective="entropic", lam=LAM, use_prev=True, steps=1600, batch=16384,
                     lr=1e-3, device=DEV, seed=s, sim=sim)
        hed.eval()
        with torch.no_grad(): W = rollout_pnl(hed, St, K, SIGMA, T, C, prem, True)
        de.append(erisk(W)); dc.append(cvar99(W)); dr.append(rmse(W))
    de, dc, dr = map(lambda x: np.array(x), (de, dc, dr))
    print(f"\n=== {mname}   (fair premium {prem:.3f}, const band h*={bh:.3f}, WW scale*={ws:.2f}) ===")
    print(f"{'strategy':<24}{'entropic':>10}{'CVaR99':>9}{'RMSE':>8}")
    print(f"{'naive delta':<24}{erisk(Wd):>10.3f}{cvar99(Wd):>9.3f}{rmse(Wd):>8.3f}")
    print(f"{'constant band':<24}{erisk(Wb):>10.3f}{cvar99(Wb):>9.3f}{rmse(Wb):>8.3f}")
    print(f"{'WW gamma-band (strong)':<24}{erisk(Www):>10.3f}{cvar99(Www):>9.3f}{rmse(Www):>8.3f}")
    print(f"{'deep (3 seeds)':<24}{de.mean():>10.3f}{dc.mean():>9.3f}{dr.mean():>8.3f}"
          f"   (CVaR99 seeds {np.round(dc,2)})")
    print(f"  deep vs WW-band (the strong classical): entropic {100*(erisk(Www)-de.mean())/abs(erisk(Www)):+.1f}%"
          f"   CVaR99 {100*(cvar99(Www)-dc.mean())/cvar99(Www):+.1f}%   RMSE {100*(rmse(Www)-dr.mean())/rmse(Www):+.1f}%")
