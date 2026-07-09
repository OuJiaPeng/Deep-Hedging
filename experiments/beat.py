"""Phase-2-equivalent WIN: with proportional transaction costs, BS delta is provably
suboptimal (re-hedges every step, over-pays cost). Deep hedger vs a STRONG, fairly-tuned
classical opponent -- the optimal no-trade band around delta (practical Whalley-Wilmott
structure) -- plus naive delta. Same objective for everyone; paired CRN evaluation.

Objective is a CLI arg so the win can be shown two ways, each individually clean:
  quadratic (mean-variance): at c=0 the deep hedge TIES delta (delta is variance-optimal),
      so any edge under cost is PURELY cost-avoidance -- no confound.  [primary, airtight]
  entropic (exp-utility / tail): delta is not tail-optimal, so deep also shapes the tail;
      part of the edge is present at c=0 (tail) and part grows with cost.  [ties to the
      entropy-regularized-control paper; report CVaR]

Usage: python scripts/beat.py [quadratic|entropic]
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, Hedger, rollout_pnl, delta_hedge_pnl,
    band_hedge_pnl, train_hedger, entropic_risk,
)

OBJ = sys.argv[1] if len(sys.argv) > 1 else "quadratic"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
S0, K, SIGMA, T, N = 100.0, 100.0, 0.20, 1.0, 30
LAM, ALPHA = 1.0, 0.99
premium = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
COSTS = [0.0, 0.005, 0.01, 0.02]

def obj_risk(W):
    """The scalar each strategy minimizes (lower = better)."""
    W = W.double().cpu()
    if OBJ == "quadratic":
        return float(torch.sqrt((W * W).mean()))              # RMSE of terminal wealth
    if OBJ == "entropic":
        return float(entropic_risk(W, LAM))
    raise ValueError(OBJ)

OBJ_NAME = {"quadratic": "RMSE", "entropic": f"entropic(l={LAM})"}[OBJ]

def summarize(W, turn):
    Wd = W.double().cpu(); L = -Wd
    return dict(obj=obj_risk(W), mean=float(Wd.mean()), std=float(Wd.std()),
                cvar99=float(torch.quantile(L, 0.99) + 100*torch.clamp(L-torch.quantile(L,0.99),min=0).mean()),
                turn=float(turn.double().cpu().mean()))

gen_val  = torch.Generator(device=DEV).manual_seed(999)
gen_test = torch.Generator(device=DEV).manual_seed(12345)
S_val  = simulate_gbm(S0, 0.0, SIGMA, T, N, 100_000, DEV, gen_val)
S_test = simulate_gbm(S0, 0.0, SIGMA, T, N, 200_000, DEV, gen_test)
H_GRID = torch.linspace(0.0, 0.40, 41)

print(f"OBJECTIVE = {OBJ} ({OBJ_NAME})   device={DEV}   ATM call sigma={SIGMA} T={T} N={N}")
print(f"premium={premium:.4f}\n")
rows = []
for c in COSTS:
    hedger = Hedger(n_features=5, hidden=(64, 64), use_prev=True).to(DEV)
    train_hedger(hedger, K=K, sigma=SIGMA, T=T, N=N, cost=c, premium=premium, S0=S0,
                 objective=OBJ, lam=LAM, alpha=ALPHA, use_prev=True,
                 steps=1600, batch=16384, lr=1e-3, device=DEV, seed=0)
    hedger.eval()
    with torch.no_grad():
        W_nn, turn_nn = rollout_pnl(hedger, S_test, K, SIGMA, T, c, premium, True, return_turnover=True)
    W_d, turn_d = delta_hedge_pnl(S_test, K, SIGMA, T, c, premium, return_turnover=True)
    if c == 0.0:
        best_h = 0.0
    else:
        best_h, best_r = 0.0, 1e18
        for h in H_GRID.tolist():
            r = obj_risk(band_hedge_pnl(S_val, K, SIGMA, T, c, premium, h))
            if r < best_r: best_r, best_h = r, h
    W_b, turn_b = band_hedge_pnl(S_test, K, SIGMA, T, c, premium, best_h, return_turnover=True)

    sd, sb, sn = summarize(W_d, turn_d), summarize(W_b, turn_b), summarize(W_nn, turn_nn)
    print(f"===== cost c = {c:.3%} =====   (tuned band width h* = {best_h:.3f})")
    print(f"{'strategy':<22}{OBJ_NAME:>14}{'E[W]':>9}{'std':>8}{'CVaR99':>9}{'turnover':>10}")
    for name, s in [("naive delta", sd), (f"tuned band(h={best_h:.2f})", sb), ("deep hedge", sn)]:
        print(f"{name:<22}{s['obj']:>14.4f}{s['mean']:>9.3f}{s['std']:>8.3f}{s['cvar99']:>9.3f}{s['turn']:>10.3f}")
    dv = 100*(sd['obj']-sn['obj'])/sd['obj']; bv = 100*(sb['obj']-sn['obj'])/sb['obj']
    print(f"  deep {OBJ_NAME} reduction:  vs delta {dv:+.1f}%  |  vs tuned band {bv:+.1f}%"
          f"   | turnover {100*(1-sn['turn']/max(sd['turn'],1e-9)):.0f}% < delta\n")
    rows.append(dict(c=c, delta=sd, band=sb, deep=sn, h=best_h))

print("="*72)
print(f"HEADLINE ({OBJ_NAME} reduction of deep hedge vs the strong tuned band):")
for r in rows:
    if r['c'] == 0:
        print(f"  c=0.00%:  deep {r['deep']['obj']:.3f} vs delta {r['delta']['obj']:.3f}"
              f"  -> ratio {r['deep']['obj']/r['delta']['obj']:.3f}  (expect ~1.00: TIE / no confound)")
        continue
    red = 100*(r['band']['obj']-r['deep']['obj'])/r['band']['obj']
    print(f"  c={r['c']:.2%}:  deep {r['deep']['obj']:.3f}  vs tuned band {r['band']['obj']:.3f}"
          f"  -> {red:+.1f}%   [CVaR99 {r['deep']['cvar99']:.2f} vs {r['band']['cvar99']:.2f};"
          f" {100*(1-r['deep']['turn']/r['delta']['turn']):.0f}% less turnover than delta]")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cs = [r['c'] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(cs, [r['delta']['obj'] for r in rows], "o-", label="naive delta")
    a1.plot(cs, [r['band']['obj'] for r in rows], "s-", label="tuned no-trade band")
    a1.plot(cs, [r['deep']['obj'] for r in rows], "^-", label="deep hedge", lw=2)
    a1.set_xlabel("proportional cost c"); a1.set_ylabel(OBJ_NAME)
    a1.set_title(f"Hedging risk ({OBJ_NAME}) vs cost"); a1.legend()
    a2.plot(cs, [r['delta']['turn'] for r in rows], "o-", label="naive delta")
    a2.plot(cs, [r['band']['turn'] for r in rows], "s-", label="tuned band")
    a2.plot(cs, [r['deep']['turn'] for r in rows], "^-", label="deep hedge", lw=2)
    a2.set_xlabel("proportional cost c"); a2.set_ylabel("mean turnover"); a2.set_title("Turnover vs cost"); a2.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", f"beat_{OBJ}.png")
    fig.savefig(out, dpi=130); print(f"\nfigure -> {out}")
except Exception as e:
    print("plot skipped:", e)
