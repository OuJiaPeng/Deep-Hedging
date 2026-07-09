"""Phase-1-equivalent VALIDATION: in a frictionless, complete Black-Scholes market the
deep hedger must RECOVER the BS delta hedge.

Two things must hold to trust the machinery before we turn on frictions:
  (A) the learned policy's terminal-wealth risk MATCHES the exact BS-delta hedge
      (both are discretization-limited; the deep hedge cannot beat delta here, by
      completeness -- delta is the variance-minimizing hedge), and
  (B) the learned hedge RATIO reproduces the analytic delta N(d1) across (moneyness, time).

Paired comparison on common random numbers (CRN), mirroring the liquidation project.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, bs_delta_call, simulate_gbm, Hedger, make_features,
    rollout_pnl, delta_hedge_pnl, train_hedger,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ---- market / contract ----
S0, K, SIGMA, T, N = 100.0, 100.0, 0.20, 1.0, 30
COST = 0.0                                   # frictionless
premium = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
print(f"device={DEV}  ATM call  S0={S0} K={K} sigma={SIGMA} T={T} N={N} cost={COST}")
print(f"BS premium p0 = {premium:.5f}")

def stats(W):
    W = W.double().cpu()
    L = -W
    return dict(mean=float(W.mean()), std=float(W.std()),
                cvar95=float(torch.quantile(L, 0.95) + 20*torch.clamp(L-torch.quantile(L,0.95),min=0).mean()),
                cvar99=float(torch.quantile(L, 0.99) + 100*torch.clamp(L-torch.quantile(L,0.99),min=0).mean()),
                p01=float(torch.quantile(W, 0.01)), p99=float(torch.quantile(W, 0.99)))

# ---- shared CRN test set ----
gen_test = torch.Generator(device=DEV).manual_seed(12345)
S_test = simulate_gbm(S0, 0.0, SIGMA, T, N, 200_000, DEV, gen_test)

# ---- baseline: exact BS delta hedge on the test paths ----
W_bs = delta_hedge_pnl(S_test, K, SIGMA, T, COST, premium)
sb = stats(W_bs)

# ---- train deep hedger (pure Markov state, no prev-holding: cleanest recovery) ----
hedger = Hedger(n_features=4, hidden=(64, 64), use_prev=False).to(DEV)
hist = train_hedger(hedger, K=K, sigma=SIGMA, T=T, N=N, cost=COST, premium=premium,
                    S0=S0, objective="quadratic", use_prev=False,
                    steps=1500, batch=16384, lr=1e-3, device=DEV, seed=0)
print(f"train loss {hist[0][1]:.5f} -> {hist[-1][1]:.5f}")

hedger.eval()
with torch.no_grad():
    W_nn = rollout_pnl(hedger, S_test, K, SIGMA, T, COST, premium, use_prev=False)
sn = stats(W_nn)

print("\n=== (A) terminal-wealth risk on 200k CRN paths (frictionless) ===")
print(f"{'strategy':<16}{'mean':>10}{'std':>10}{'CVaR95(L)':>12}{'CVaR99(L)':>12}{'p01(W)':>10}")
for name, s in [("BS delta (exact)", sb), ("deep hedge", sn)]:
    print(f"{name:<16}{s['mean']:>10.4f}{s['std']:>10.4f}{s['cvar95']:>12.4f}{s['cvar99']:>12.4f}{s['p01']:>10.4f}")
print(f"\nstd ratio  deep/BS = {sn['std']/sb['std']:.4f}   (recovery => ~1.00; deep CANNOT beat delta here)")
print(f"paired mean |W_nn - W_bs| = {float((W_nn-W_bs).abs().mean()):.4f}  (both track the same replication error)")

# ---- (B) learned hedge ratio vs analytic delta on a (moneyness, time) grid ----
print("\n=== (B) learned delta vs analytic N(d1) across the state grid ===")
errs = []
for tau_frac in (0.9, 0.5, 0.2, 0.05):
    tau = tau_frac * T
    S_grid = torch.linspace(80, 120, 41, device=DEV, dtype=torch.float64)
    taus = torch.full_like(S_grid, tau)
    feats = make_features(S_grid, K, taus, SIGMA, T,
                          torch.zeros_like(S_grid), use_prev=False)
    with torch.no_grad():
        d_nn = hedger(feats.to(torch.float32)).double().cpu().numpy()
    d_bs = bs_delta_call(S_grid, K, tau, SIGMA).cpu().numpy()
    mae = float(np.mean(np.abs(d_nn - d_bs)))
    errs.append(mae)
    print(f"  tau={tau_frac:>4.2f}T : mean|delta_nn - N(d1)| = {mae:.4f}   "
          f"(at ATM: nn={np.interp(100,S_grid.cpu().numpy(),d_nn):.3f} vs bs={np.interp(100,S_grid.cpu().numpy(),d_bs):.3f})")
print(f"  grid-mean delta MAE = {np.mean(errs):.4f}")

# ---- save learned-vs-analytic delta figure ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Sg = torch.linspace(80, 120, 81, device=DEV, dtype=torch.float64)
    for tau_frac, col in zip((0.9, 0.5, 0.2, 0.05), ["#1f77b4","#2ca02c","#ff7f0e","#d62728"]):
        tau = tau_frac*T; taus = torch.full_like(Sg, tau)
        feats = make_features(Sg, K, taus, SIGMA, T, torch.zeros_like(Sg), use_prev=False)
        with torch.no_grad():
            dnn = hedger(feats.to(torch.float32)).double().cpu().numpy()
        dbs = bs_delta_call(Sg, K, tau, SIGMA).cpu().numpy()
        ax.plot(Sg.cpu().numpy(), dbs, color=col, lw=2, alpha=0.5, label=f"BS N(d1), tau={tau_frac}T")
        ax.plot(Sg.cpu().numpy(), dnn, color=col, lw=1.2, ls="--")
    ax.set_xlabel("spot S"); ax.set_ylabel("hedge ratio (shares)")
    ax.set_title("Recovery: deep hedge (dashed) vs analytic BS delta (solid)")
    ax.legend(fontsize=7, loc="upper left"); fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "recover_delta.png")
    fig.savefig(out, dpi=130); print(f"\nfigure -> {out}")
except Exception as e:
    print("plot skipped:", e)
