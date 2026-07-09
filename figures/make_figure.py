"""The corrected deep-hedging result in one figure -> figures/summary.png

Two panels tell the honest story for a short European call under 1% proportional cost:
  A. mean-variance objective: the deep hedge TIES the correctly-implemented no-trade band
     (the band is the optimum there, so this is a boundary, not a win).
  B. tail (entropic) objective: the deep hedge nearly reaches the exact -- but intractable --
     Hodges-Neuberger optimum, while the DEPLOYABLE formulas leave a 37-56% gap.

Numbers are the corrected, post-audit results; reproduce them with:
  scripts/multiseed.py (panel A, RMSE)  ·  scripts/win_test.py + scripts/hn_dp.py (panel B, entropic).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Panel A: mean-variance (RMSE of terminal wealth), c=1%  [scripts/multiseed.py] ----
A_names = ["naive\ndelta", "tuned\nband", "deep\nhedge"]
A_rmse = [3.209, 2.657, 2.681]

# ---- Panel B: tail objective (entropic risk), c=1%  [win_test.py / hn_dp.py] ----
HN = 3.296                                   # exact Hodges-Neuberger optimum (DP) -- the floor
B_names = ["naive delta", "WW gamma-band", "constant band", "deep hedge"]
B_ent = [5.694, 5.139, 4.508, 3.510]
B_gap = [100 * (v - HN) / HN for v in B_ent]  # % above the exact optimum

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, (a, bx) = plt.subplots(1, 2, figsize=(11, 4.4))

# Panel A
colsA = ["#8c8c8c", "#2ca02c", "#d62728"]
a.bar(A_names, A_rmse, color=colsA, width=0.6)
for i, v in enumerate(A_rmse):
    a.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
a.set_ylabel("RMSE of terminal wealth  (lower = better)")
a.set_ylim(0, 3.7)
a.set_title("A.  Mean-variance: deep TIES the classical optimum", fontsize=11)
a.annotate("deep ≈ band\n(the band IS optimal here → a boundary)",
           xy=(2, 2.681), xytext=(0.55, 3.25), fontsize=8.5, color="#333",
           arrowprops=dict(arrowstyle="->", color="#888"))

# Panel B — horizontal bars, HN optimum as the floor line
y = np.arange(len(B_names))[::-1]
colsB = ["#8c8c8c", "#8c8c8c", "#8c8c8c", "#d62728"]
bx.barh(y, B_ent, color=colsB, height=0.6)
bx.axvline(HN, color="#2ca02c", ls="--", lw=2)
bx.text(HN, len(B_names) - 0.35, f"  exact optimum\n  (HN, intractable) = {HN:.2f}",
        color="#2ca02c", fontsize=8.5, va="top")
for yi, v, g in zip(y, B_ent, B_gap):
    bx.text(v + 0.08, yi, f"{v:.2f}   (+{g:.0f}% vs optimum)", va="center", fontsize=8.5)
bx.set_yticks(y); bx.set_yticklabels(B_names, fontsize=9)
bx.set_xlim(0, 7.2); bx.set_xlabel("entropic (tail) risk  (lower = better)")
bx.set_title("B.  Tail risk: deep ~6.5% off optimum; formulas 37–56% off", fontsize=11)

fig.suptitle("Deep hedging a short call under 1% cost — matches the optimum where one exists, "
             "captures the intractable one on tail risk", fontsize=11.5, y=1.02)
fig.tight_layout()
out_dir = os.path.join(ROOT, "figures"); os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "summary.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
