"""First real-data chapter: sim-to-real transfer test on actual price history.

Train the net on GENERATED GBM paths (sigma calibrated on the EARLY era only), then hedge REAL
held-out windows from the LATE era, paired against BS delta and the tuned no-trade band. Answers:
does a sim-trained hedge survive contact with real prices, on the same paths as the classics?

Honest setup: sigma from train era only (no look-ahead); band tuned on generated validation (not on
the real test); each real window normalized to start at 100 (hedging is scale-free -> reuse K=100 code);
mu=0 in training (delta hedging is drift-agnostic; real drift hits every strategy equally on a paired
path). 1-month options (N=21 trading days), daily rebalance, 10 bps proportional cost.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, Hedger, rollout_pnl, delta_hedge_pnl,
    band_hedge_pnl, train_hedger, entropic_risk, fetch,
)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, K, C, LAM = 21, 100.0, 0.001, 1.0          # 1-month option, 10 bps cost
T = N / 252.0

def make_episodes(close, step=1):
    """Rolling windows of N+1 closes, each normalized to start at 100. -> [n, N+1]."""
    W = []
    for i in range(0, len(close) - N, step):
        w = close[i:i + N + 1]
        W.append(w / w[0] * 100.0)
    return np.array(W)

def cvar99(W): W = np.asarray(W); L = -W; q = np.quantile(L, 0.99); return q + 100 * np.clip(L - q, 0, None).mean()
def erisk_np(W): return np.log(np.mean(np.exp(-LAM * np.asarray(W)))) / LAM

H_GRID = torch.linspace(0.0, 0.4, 41)
for sym in ["SPY", "AAPL"]:
    close = fetch(sym)
    logret = np.diff(np.log(close))
    split = int(0.6 * len(close))                                  # ~15y train / ~10y test
    sig = float(np.std(logret[:split]) * math.sqrt(252))           # calibrate on EARLY era only
    prem = float(bs_price_call(torch.tensor(100.0), K, T, sig))
    # real held-out test episodes (LATE era), non-overlapping (independent) + all-rolling (more, correlated)
    ep_all = make_episodes(close[split:], step=1)
    ep_indep = make_episodes(close[split:], step=N)
    St = torch.tensor(ep_all, device=DEV, dtype=torch.float64)
    print(f"\n===== {sym}  (sigma_train={sig:.3f}, {len(ep_indep)} independent / {len(ep_all)} rolling test episodes, "
          f"cost={C:.1%}) =====")

    # tune band on GENERATED validation at sigma_train (no real-data peeking)
    gv = torch.Generator(device=DEV).manual_seed(999)
    Sv = simulate_gbm(100.0, 0.0, sig, T, N, 80_000, DEV, gv)
    def erisk_t(W): return float(entropic_risk(W.double().cpu(), LAM))

    for obj in ["quadratic", "entropic"]:
        torch.manual_seed(0)
        hed = Hedger(n_features=5, hidden=(64, 64), use_prev=True).to(DEV)
        train_hedger(hed, K=K, sigma=sig, T=T, N=N, cost=C, premium=prem, S0=100.0,
                     objective=obj, lam=LAM, use_prev=True, steps=1500, batch=8192, lr=1e-3, device=DEV, seed=0)
        hed.eval()
        score = erisk_t if obj == "entropic" else (lambda W: float(torch.sqrt((W.double().cpu()**2).mean())))
        bh = min(H_GRID.tolist(), key=lambda h: score(band_hedge_pnl(Sv, K, sig, T, C, prem, h)))
        # evaluate on REAL test episodes, paired
        Wd = delta_hedge_pnl(St, K, sig, T, C, prem).double().cpu().numpy()
        Wb = band_hedge_pnl(St, K, sig, T, C, prem, bh).double().cpu().numpy()
        with torch.no_grad():
            Wn = rollout_pnl(hed, St, K, sig, T, C, prem, True).double().cpu().numpy()
        print(f"  [{obj}] band h*={bh:.3f}   (on REAL {sym} test paths, cost per episode, lower=better)")
        print(f"    {'strategy':<12}{'mean cost':>11}{'std':>8}{'CVaR99':>9}{'entropic':>10}")
        for nm, W in [("BS delta", Wd), ("tuned band", Wb), ("deep net", Wn)]:
            print(f"    {nm:<12}{-W.mean():>11.3f}{W.std():>8.3f}{cvar99(W):>9.3f}{erisk_np(W):>10.3f}")
        m = "entropic" if obj == "entropic" else "RMSE"
        deep_metric = erisk_np(Wn) if obj == "entropic" else math.sqrt((Wn**2).mean())
        band_metric = erisk_np(Wb) if obj == "entropic" else math.sqrt((Wb**2).mean())
        print(f"    deep vs band ({m}): {100*(band_metric-deep_metric)/abs(band_metric):+.1f}%   "
              f"CVaR99 {100*(cvar99(Wb)-cvar99(Wn))/cvar99(Wb):+.1f}%")
