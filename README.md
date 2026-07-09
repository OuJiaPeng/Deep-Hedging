# Deep Hedging

> A neural network learns to hedge an option by backpropagating through a differentiable market simulator.
> It is judged against the strongest classical baselines available. In frictionless Black-Scholes it recovers
> delta; under transaction costs and mean-variance risk it ties the corrected no-trade band. The real edge is
> tail risk: in GBM, the learned hedge lands within **6.5%** of the exact Hodges-Neuberger DP optimum, while
> deployable bands sit **37-56%** above it. Under Merton jumps, where the DP oracle is not reliable, it still
> beats Whalley-Wilmott by **16.5% CVaR99**. No RL loop, no PPO: just pathwise gradients.

Companion to [RL Optimal Liquidation](https://github.com/OuJiaPeng/RL-Optimal-Liquidation). That project is
the boundary case: learned control ties a deployable classical optimum. This one is the complementary case:
on tail risk, deployable formulas leave a large gap to the exact-but-unscalable optimum, and the learned
policy nearly closes it.

![Summary: deep hedge ties the classical optimum on mean-variance, and nearly reaches the intractable tail optimum](figures/summary.png)

*One-figure version (`python figures/make_figure.py`): **(A)** on mean-variance risk, deep hedging ties the
correct edge-reflecting no-trade band; **(B)** on tail risk, it gets close to the exact Hodges-Neuberger
optimum while deployable bands remain far above it.*

## Results

All numbers are post-audit; an adversarial review found and fixed a baseline bug. The full account is in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

| Phase | Result |
|---|---|
| **recover** (frictionless) | deep = BS delta: std ratio **1.009**, hedge-ratio MAE **0.008**. Validation. |
| **tie** (cost, mean-variance) | deep = correct edge-reflecting no-trade band, ±1% over 5 seeds. The band is the optimum here, so deep can only match it — a boundary. Beats naive delta +6/+16/+30%. |
| **win** (cost, tail objective) | deep within **6.5%** of the exact Hodges–Neuberger optimum (DP-certified); deployable bands **37–56%** above it. Same pattern under Merton jumps (**+16.5% CVaR₉₉** over Whalley–Wilmott). |

## Claim Boundary

The certified optimum comparison is for GBM, where the Hodges-Neuberger DP is stable enough to use as an
oracle. Merton jumps and real-data tests are empirical comparisons against deployable baselines; the jump DP
is not treated as a valid oracle.

## Formulation

| | |
|---|---|
| **Contract** | short one European call, K=100, σ=0.2, T=1, r=0, N=30 dates (extensible; Asian variant in `asian_transformer.py`) |
| **State** | standardized moneyness log(S/K)/(σ√τ), moneyness, time-to-maturity, current holding |
| **Action** | stock holding δ each step |
| **Objective** | mean-variance / **entropic** (exp-utility) / CVaR |
| **Training** | pathwise gradient through a differentiable GBM/Merton simulator (Adam, cosine LR), GPU |
| **Baselines** | exact BS delta · Leland · edge-reflecting no-trade band · Whalley–Wilmott gamma-band · **exact Hodges–Neuberger optimum (DP)** · paired CRN eval, 200k paths + test-path bootstrap |

## Quickstart

Requires Python with PyTorch, NumPy, and Matplotlib. CUDA is recommended; the experiment scripts fall back to
CPU, but the training runs are slow there. These are research runs, not a unit-test suite.

```bash
python experiments/recover.py           # frictionless recovery of BS delta
python experiments/multiseed.py         # mean-variance: deep ties the correct band (boundary)
python experiments/win_test.py          # tail objective: deep vs constant & WW bands, GBM + jumps
python experiments/hn_dp.py             # exact Hodges-Neuberger optimum (DP) — certifies the gap (GBM)
python experiments/robustness.py        # vol misspecification + jumps
python experiments/asian_transformer.py # transformer vs FF on a path-dependent Asian option
python experiments/real_data_test.py    # sim-to-real smoke test; downloads/caches Yahoo closes on first run
```

Full account, mechanism, the audit correction, and how to defend it: [`PROJECT_STATUS.md`](PROJECT_STATUS.md),
[`INTERVIEW_NOTES.md`](INTERVIEW_NOTES.md).
