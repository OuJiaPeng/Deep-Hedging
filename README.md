# Deep Hedging

A small research repo on neural option hedging.

The setup is plain: you are short one European call. At each hedge date, a policy chooses how much stock to
hold. At expiry, it pays the option payoff. The neural policy trains by backpropagating through simulated
price paths, transaction costs included.

The point is not "deep learning beats Black-Scholes." It should not. The point is to find the line between
problems where classical hedges already solve the job and problems where a learned hedge can help.

## Short Version

- In frictionless Black-Scholes, the network recovers the Black-Scholes delta hedge.
- With transaction costs and a mean-variance objective, it ties the correct no-trade band.
- With a tail-risk objective, the learned hedge gets close to the exact Hodges-Neuberger dynamic program in
  GBM, while practical band hedges stay much farther away.
- Under Merton jumps, the dynamic-programming oracle is not reliable, so the claim is simpler: the learned
  hedge beats the deployable bands on matched simulated paths.

![Summary: deep hedge ties the classical optimum on mean-variance, and nearly reaches the tail-risk optimum](figures/summary.png)

## What Gets Tested

The project has three passes.

| Pass | Check | Result |
|---|---|---|
| Recover | Frictionless Black-Scholes | Deep hedge matches delta: std ratio **1.009**, hedge-ratio MAE **0.008**. |
| Tie | Transaction costs, mean-variance risk | Deep hedge ties the edge-reflecting no-trade band across costs. |
| Win | Transaction costs, tail risk | In GBM, deep is **6.5%** above the exact HN optimum; deployable bands are **37-56%** above it. |

The jump result is empirical rather than oracle-certified: under Merton jumps, deep beats Whalley-Wilmott by
**16.5% CVaR99**, but the HN dynamic program is not used as a floor there.

## How It Works

The model is a feedforward hedge policy:

```text
state features -> stock holding
```

The state includes moneyness, time to maturity, a near-expiry standardized moneyness feature, and current
holding when trading costs matter. The rollout is differentiable end to end, so training is ordinary
gradient descent through simulated P&L. This is not PPO or policy-gradient RL.

Main files:

- `deep_hedging/data.py`: GBM, Merton jumps, and cached real-data fetches.
- `deep_hedging/baselines.py`: Black-Scholes delta, Leland, no-trade bands, Whalley-Wilmott.
- `deep_hedging/model.py`: features, neural hedge, differentiable rollout, training loop.
- `deep_hedging/risk.py`: mean-square, entropic risk, and CVaR losses.
- `experiments/hn_dp.py`: Hodges-Neuberger dynamic program for the clean GBM oracle.

## Baselines

The comparison ladder is deliberately strong.

- **Black-Scholes delta:** the frictionless benchmark.
- **Leland delta:** a simple transaction-cost adjustment.
- **No-trade band:** hold while your position stays inside a band around delta.
- **Whalley-Wilmott band:** a state-dependent gamma-scaled band.
- **Hodges-Neuberger DP:** the exact entropic-risk optimum in the clean GBM setting.

One implementation detail matters: the no-trade band reflects to the nearest band edge. Rehedging to the
center overtrades and is not the right baseline.

## Claim Boundary

The certified optimum comparison is for GBM, where the HN dynamic program behaves well enough to use as an
oracle. Merton jumps and real-data tests are comparisons against practical baselines, not claims of exact
optimality.

## Quickstart

Requires Python with PyTorch, NumPy, and Matplotlib. CUDA is recommended. The scripts are research runs, not a
unit-test suite, so several of them take a while.

```bash
python experiments/recover.py           # recover Black-Scholes delta
python experiments/multiseed.py         # transaction costs: deep ties the no-trade band
python experiments/win_test.py          # tail risk: deep vs constant and WW bands
python experiments/hn_dp.py             # exact HN optimum for GBM
python experiments/robustness.py        # vol misspecification and jumps
python experiments/bootstrap_ci.py      # paired bootstrap on the tail-risk gap
python experiments/asian_transformer.py # path-dependent Asian option
python experiments/real_data_test.py    # sim-to-real smoke test on SPY/AAPL
```

For the longer account, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
