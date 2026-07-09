# Deep Hedging - Project Status

This is the longer account of the project. The README is the skim.

## Short Version

This repo trains a neural hedge for a short European call. The policy learns by pathwise gradients through a
differentiable market simulator. No closed-form hedge is fed to the model.

The result is simple:

- In the frictionless Black-Scholes setting, the model recovers delta.
- With transaction costs and mean-variance risk, the model ties the correct no-trade band.
- With tail risk, the model gets close to the exact Hodges-Neuberger optimum in GBM, while practical band
  hedges leave a large gap.
- With jumps and real data, the comparison is empirical. The exact DP is not treated as an oracle there.

That is the useful boundary. Deep hedging is not a reason to throw away classical hedges. It helps when the
classical hedge that can actually be deployed is no longer close to the risk objective.

## Setup

The contract is one short European call:

```text
S0 = K = 100
sigma = 0.20
T = 1
r = 0
N = 30 hedge dates
```

At each date, the policy chooses a stock holding. Terminal wealth is:

```text
premium
+ hedge P&L
- trading costs
- option payoff
- final liquidation cost
```

The training loss is one of:

- mean-square terminal wealth,
- entropic risk,
- CVaR.

The simulator uses GBM for the certified result and Merton jumps for the incomplete-market stress test.

## Results

### 1. Recover Black-Scholes Delta

In the frictionless GBM case, the correct answer is the Black-Scholes delta hedge. The network should not beat
it. It should recover it.

`experiments/recover.py` does that:

- terminal-wealth std: deep **1.256** vs delta **1.244**,
- std ratio: **1.009**,
- learned hedge-ratio MAE vs analytic delta: **0.008**.

This validates the simulator, rollout, features, and optimizer.

### 2. Tie The No-Trade Band Under Costs

With proportional transaction costs and a mean-variance objective, the right classical object is a no-trade
band around delta. You hold your current position while it stays inside the band. If it leaves, you trade only
to the nearest band edge.

That edge-reflection detail matters. Trading back to the center overtrades and makes the baseline too weak.

With the corrected band, deep hedging ties the band across costs:

| cost | naive delta | Leland | tuned band | deep hedge |
|---:|---:|---:|---:|---:|
| 0.5% | 1.952 | 1.902 | 1.833 | 1.834 |
| 1.0% | 3.209 | 3.028 | 2.657 | 2.681 |
| 2.0% | 5.995 | 5.410 | 4.213 | 4.217 |

Lower is better. The lesson is not that deep beats the band. It does not. The lesson is that the learned
policy reaches the right classical answer when that answer already matches the objective.

### 3. Beat Practical Bands On Tail Risk

Tail risk changes the problem. A symmetric band around Black-Scholes delta is built for a variance-style hedge.
An entropic or CVaR objective cares much more about bad downside outcomes. The best hedge can shift and become
asymmetric.

In GBM with 1% transaction cost and entropic risk:

| strategy | entropic risk | gap above HN optimum | CVaR99 |
|---|---:|---:|---:|
| exact HN optimum | **3.296** | - | - |
| deep hedge | 3.510 | **+6.5%** | 6.21 |
| constant band | 4.508 | +37% | 7.55 |
| Whalley-Wilmott band | 5.139 | +56% | 7.61 |

The HN number comes from `experiments/hn_dp.py`. The deep hedge does not beat the optimum. It gets close to an
optimum that is only practical here because the toy state space is small. The deployable bands are much farther
away.

The paired bootstrap in `experiments/bootstrap_ci.py` checks that the tail gap is not just test-path noise.

### 4. Jumps

Under Merton jumps, the market is incomplete and the tail objective becomes harder. The same pattern appears:
the learned hedge beats the Whalley-Wilmott band by about **16.5% CVaR99** and **42.6% entropic risk**.

The jump HN dynamic program is not reliable. In this setting the exponential-utility value function is too
stiff for the simple grid/quadrature DP used here, and it returns a value worse than an achieved policy. An
optimum cannot do that. So the jump result is stated as a matched empirical comparison against deployable
baselines, not as a certified optimum gap.

### 5. Robustness And Path Dependence

`experiments/robustness.py` checks volatility misspecification and jumps under the mean-variance objective.
The main pattern holds: deep hedging behaves like the band when the band is the right tool, and tail metrics
improve when the objective rewards tail protection.

`experiments/asian_transformer.py` tests a path-dependent Asian option. A myopic feedforward hedge that only
sees spot and time is missing information. An informed feedforward model with the running average improves,
and a causal transformer over price history performs best in the run here. The caveat is important: with the
right sufficient statistic, a feedforward model should in principle be able to match the transformer. The
transformer result is an architecture/optimization result, not proof that sequence models have extra
information.

## Real-Data Smoke Test

`experiments/real_data_test.py` trains in simulation and hedges real SPY/AAPL price windows out of sample. This
is a transfer check, not the certified result.

The simulated hedge does not collapse on real data. On the entropic objective it beats the tuned band in the
single-name tests. The sharp simulated CVaR99 win is harder to measure in real data because one name gives too
few independent monthly episodes, and real crashes contain jumps that a GBM-only generator does not train on.

A full real-data version would need a wider universe and a richer generator with jumps and stochastic or rough
volatility.

## Method Notes

- Training is pathwise gradient descent through simulated P&L, not PPO.
- Evaluation uses common random numbers, so policies see the same test paths.
- The no-trade band baseline reflects to the nearest band edge.
- Training-seed spread is not a confidence interval. Bootstrap CIs are computed separately where they matter.
- The HN DP is an oracle for clean GBM only.

## Reproduce

```bash
python experiments/recover.py
python experiments/multiseed.py
python experiments/win_test.py
python experiments/hn_dp.py
python experiments/robustness.py
python experiments/bootstrap_ci.py
python experiments/asian_transformer.py
python experiments/real_data_test.py
```
