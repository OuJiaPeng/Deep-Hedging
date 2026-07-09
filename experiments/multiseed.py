"""Multi-seed the vanilla win (5 seeds) for confidence intervals, and complete the
classical ladder with Leland's modified-volatility delta. Mean-variance (RMSE) objective.

Locks the claim: is the deep hedge's edge over the tuned no-trade band positive on EVERY
seed, and how tight is the interval?
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, Hedger, rollout_pnl, delta_hedge_pnl,
    band_hedge_pnl, train_hedger, leland_vol,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S0, K, SIGMA, T, N = 100.0, 100.0, 0.20, 1.0, 30
premium = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
COSTS = [0.005, 0.01, 0.02]
SEEDS = [0, 1, 2, 3, 4]
dt = T / N

def rmse(W): W = W.double().cpu(); return float(torch.sqrt((W*W).mean()))

gen_val  = torch.Generator(device=DEV).manual_seed(999)
gen_test = torch.Generator(device=DEV).manual_seed(12345)
S_val  = simulate_gbm(S0, 0.0, SIGMA, T, N, 100_000, DEV, gen_val)
S_test = simulate_gbm(S0, 0.0, SIGMA, T, N, 200_000, DEV, gen_test)
H_GRID = torch.linspace(0.0, 0.40, 41)

print(f"multi-seed vanilla win (RMSE, mean-variance)  seeds={SEEDS}  device={DEV}")
print(f"{'cost':>6} | {'delta':>7} {'leland':>7} {'band':>7} | {'deep mean+-sd':>16} "
      f"{'win vs band %':>16} {'min seed':>9}")
summary = []
for c in COSTS:
    d = rmse(delta_hedge_pnl(S_test, K, SIGMA, T, c, premium))
    lel = rmse(delta_hedge_pnl(S_test, K, SIGMA, T, c, premium, vol_hedge=leland_vol(SIGMA, c, dt)))
    best_h, best_r = 0.0, 1e18
    for h in H_GRID.tolist():
        r = rmse(band_hedge_pnl(S_val, K, SIGMA, T, c, premium, h))
        if r < best_r: best_r, best_h = r, h
    band = rmse(band_hedge_pnl(S_test, K, SIGMA, T, c, premium, best_h))
    deeps = []
    for s in SEEDS:
        torch.manual_seed(s)
        hed = Hedger(n_features=5, hidden=(64, 64), use_prev=True).to(DEV)
        train_hedger(hed, K=K, sigma=SIGMA, T=T, N=N, cost=c, premium=premium, S0=S0,
                     objective="quadratic", use_prev=True, steps=1600, batch=16384,
                     lr=1e-3, device=DEV, seed=s)
        hed.eval()
        with torch.no_grad():
            deeps.append(rmse(rollout_pnl(hed, S_test, K, SIGMA, T, c, premium, True)))
    deeps = np.array(deeps)
    wins = 100*(band - deeps)/band
    print(f"{c:>6.1%} | {d:>7.3f} {lel:>7.3f} {band:>7.3f} | {deeps.mean():>7.3f}+-{deeps.std():>5.3f} "
          f"| {wins.mean():>7.1f}+-{wins.std():>4.1f}% {wins.min():>8.1f}%")
    summary.append((c, d, lel, band, deeps, wins))

print("\nAll-seed check (win vs band > 0 on every seed):",
      all((band - deeps > 0).all() for c, d, lel, band, deeps, wins in summary))
print("Leland vs naive delta:",
      " ".join(f"c={c:.1%}:{'better' if lel<d else 'worse'}" for c,d,lel,band,deeps,wins in summary))
