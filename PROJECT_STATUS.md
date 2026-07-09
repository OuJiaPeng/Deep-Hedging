# Deep Hedging — Project Status

**Single source of truth** for status, results, and reasoning. Companion to the RL-Optimal-Liquidation
project (same discipline: exact classical baselines, recover-then-degrade arc, paired common-random-number
evaluation, adversarial audit). All numbers below are **post-audit**: an adversarial review found and I
fixed a baseline bug that had inflated an earlier headline (see [The audit](#the-audit-what-changed)).

## Thesis (what the corrected results actually support)

**Deep hedging — a model-free neural hedge trained by pathwise gradient through a differentiable market —
recovers the classical optimum wherever one exists, and delivers real excess over the *deployable* classical
hedges precisely where the classical formulas' assumptions break: on an asymmetric tail-risk objective under
frictions.** It does not beat the exact optimum (nothing can); its value is *practical* — it lands within
~6.5% of an intractable exact optimum while the hedges a desk actually deploys leave a 37–56% gap.

This is the same backbone as the liquidation project (learned control ties the classical optimum where one is
computable) with the extra half it was missing: **where and why learning starts to beat the deployable
classical method.** Answer: not on mean-variance (the no-trade band is already optimal there), but on tail
risk, where the band's symmetry-around-delta is structurally suboptimal.

## Setup

Short one European call; GBM (or Merton jumps) with S0=K=100, σ=0.20, T=1, r=0, N=30 hedge dates;
proportional cost c on traded notional. Hedge = neural net (state → shares), trained by backprop through the
simulated P&L — no PPO, no RL, no closed form. Risk measures: quadratic (mean-variance), **entropic**
(exp-utility, = the entropy-regularized-control lens), CVaR. Paired CRN evaluation, 200k test paths.

## Results

### Phase 1 — recover (frictionless; delta is optimal)

Deep hedge reproduces the Black–Scholes delta hedge and, by market completeness, cannot beat it:
terminal-wealth std **1.256 vs exact delta 1.244 (ratio 1.009)**; learned hedge ratio vs analytic N(d₁)
grid-mean MAE **0.008**. Validation passed. (`experiments/recover.py`)

### Phase 2 — mean-variance objective under cost → **a boundary (deep ties the classical optimum)**

Classical ladder: naive delta → Leland modified-vol delta → **tuned no-trade band** (Davis–Norman /
Whalley–Wilmott edge-reflecting singular control). 5-seed deep hedge, RMSE of terminal wealth:

| cost c | naive delta | Leland | tuned band | deep hedge (5 seeds) | deep vs band |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 1.952 | 1.902 | 1.833 | 1.834 ± 0.004 | **−0.0%** |
| 1.0% | 3.209 | 3.028 | 2.657 | 2.681 ± 0.015 | **−0.9%** |
| 2.0% | 5.995 | 5.410 | 4.213 | 4.217 ± 0.023 | **−0.1%** |

Deep **ties** the correct band (no seed positive) and crushes naive delta (+6/+16/+30%). Correct: under
proportional cost with a variance objective the no-trade band **is** the optimum, so — exactly like CE-AC in
the liquidation project — learning can only match it. A boundary, not a win. (`experiments/multiseed.py`)

### Phase 3 — tail-risk objective under cost → **the practical win**

Under the **entropic / CVaR** objective (what real hedgers care about — don't blow up), the deep hedge beats
the deployable classical bands, and the exact Hodges–Neuberger optimum (DP) certifies how much room the bands
leave. GBM, c=1%, entropic λ=1, 3 seeds:

| strategy | entropic risk | gap above exact optimum | CVaR₉₉ |
|---|---:|---:|---:|
| **exact HN optimum (DP)** | **3.296** | — (the floor) | — |
| **deep hedge** | 3.510 | **+6.5%** | 6.21 |
| constant no-trade band | 4.508 | +37% | 7.55 |
| Whalley–Wilmott band | 5.139 | +56% | 7.61 |

The deep hedge lands **within 6.5% of the exact optimum**; the deployable bands leave **37–56%**. The deep
hedge is *worse* on RMSE (it correctly trades variance for tail protection). Mechanism: the bands are centered
on the BS delta (the *variance* hedge) and symmetric; the tail-optimal policy is shifted/asymmetric — HN and
the neural net can sit there, a symmetric band cannot. **DP validated** by the c=0 sanity: ρ_HN(c=0)=0.884,
just below the deep hedge's 0.909, as an optimum must be. (`experiments/win_test.py`, `experiments/hn_dp.py`)

The gap over the bands is **statistically significant**, not test-sampling noise — a paired bootstrap (1000
resamples of the 200k test paths, policies fixed) gives deep-vs-WW-band **CVaR₉₉ +1.40, 95% CI [+1.35, +1.44]**
and deep-vs-const-band CVaR₉₉ +1.33 [+1.29, +1.37]. (`experiments/bootstrap_ci.py`; supersedes the training-seed
"±", which was not a real CI — audit F4.)

**Jumps (Merton, incomplete market):** same pattern empirically — deep beats the WW band by **+16.5% CVaR₉₉ /
+42.6% entropic** (3-seed). The exact HN optimum is *not* reliably computable here (exp-utility DP is
numerically unstable under fat tails — it returned a value above an achieved policy, which is impossible), so
the jump win rests on the empirical band comparison, not a certified optimum. Stated honestly as such.

### Robustness (corrected band)

- **Vol misspecification** (calibrated σ=0.20, tested 0.15–0.30, mean-variance): deep ≈ band at/above
  calibration (−0.1% to +1.8%); +11.4% at σ=0.15, though both trail naive delta there. Edge is robust, small.
- **Jumps, mean-variance:** deep ties the band on RMSE, +6.1% CVaR₉₉ — consistent with the tail story.
  (`experiments/robustness.py`)

### Transformer + path-dependent option (Asian)

Arithmetic-Asian call — payoff `(mean(S_1..S_N)-K)^+` depends on the *whole path*, so a Markov hedger
(spot+time only) is structurally insufficient. Capacity-matched architectures (~17.5k params each), z_tau
feature, 3 seeds, mean-variance:

| architecture | RMSE c=0 | RMSE c=1% |
|---|---:|---:|
| myopic FF (spot, time) | 1.061 | 1.850 |
| informed FF (+ running average) | 0.983 | 1.834 |
| **causal transformer (raw history)** | **0.891** | **1.755** |

The path-blind myopic hedger is worst (**+16% at c=0**) — path information is essential — and the causal
transformer over raw price history is the best architecture, i.e. a sequence model earns its place on a
path-dependent payoff. Honest caveat: at c=0 the informed FF holds the Markov-sufficient statistic (running
average) and should in principle *match* the transformer; the residual gap is optimization quality, not
information, and is reported as such rather than claimed as a transformer information advantage.
(`experiments/asian_transformer.py`)

## The audit — what changed

An adversarial audit (5 reviewers reading the code) found the results were leakage-free and the math correct,
but caught three real issues, all fixed:
- **F1 (fatal, fixed):** the no-trade band rehedged to the delta *center* instead of reflecting to the band
  *boundary* — over-trading that handicapped the baseline. An earlier "+4–8% beat the band" headline was an
  artifact; corrected, deep **ties** the band on mean-variance. This is why the thesis now centers on the tail
  objective, where the win survives against the *correct* band and is certified by the HN DP.
- **F3 (fixed):** an earlier "transformer beats the informed FF by 14%" was impossible in principle (the FF had
  the sufficient statistic); it was under-capacity / under-trained / single-seed / missing a feature. Re-run
  fairly.
- **F4 (fixed):** 5-seed "±" is training-seed spread on one test draw, now labeled as such; use ddof=1.

## Why this is a win where liquidation was a boundary

Liquidation proved the deployable classical method (CE-AC) *was* the optimum, so RL added nothing. Hedging on
a tail objective is different: the exact optimum (HN) is an intractable DP, and the *deployable* formulas
(WW, no-trade bands) are far from it (37–56%). A single scalable learned policy nearly closes that gap. The
win is not "beats the theoretical optimum" — it's "delivers near-optimal tail hedging where the optimum is
intractable and the deployable formulas fall short."

## Real-data deployment (proof-of-concept + how the full version would go)

The *certified* contribution is the simulated result above — the only setting where the exact optimum exists
to measure against. Real-market deployment is **scoped and proof-of-concept'd, not claimed as a core result.**

**Proof-of-concept (actually run, `experiments/real_data_test.py`):** the net was trained entirely in the
simulator (σ calibrated on 2000–2015 *only*), then used to hedge real out-of-sample **SPY and AAPL** 1-month
options over 2015–2025 (daily rebalance, 10 bps), paired against BS delta and the tuned band. Findings:
- **Transfer holds** — the sim-trained hedge does not break on real prices; it is competitive-to-winning vs
  both classical hedges out-of-sample.
- **It wins on the metric it optimizes:** on the entropic (tail-averse) objective it beats the tuned band by
  **+9% (AAPL) to +26% (SPY)** on real held-out paths.
- **The sharp CVaR₉₉ tail-win from the sim does *not* cleanly reproduce**, for two honest reasons: (i) one
  name yields ~119 *independent* monthly windows, so a 1%-tail is decided by ~2 real crashes — statistically
  unresolvable; (ii) real crashes are **jumps the GBM-trained net never saw**.

**The full version (not built — the shape):**
1. **Enrich the generator** — GBM + jumps + rough (fBm-in-vol) volatility, parameters randomized
   (domain randomization) — so the net trains on crash-like, long-memory dynamics, not smooth GBM. (Rough vol
   also breaks the Markov property, which is a *second* place the net should beat the Markov classical hedges.)
2. **Go wide** — ~100–500 liquid names, short-dated options → tens of thousands of independent episodes,
   enough to actually *measure* the tail rather than infer it.
3. Train on the generator (calibrated per-era), **test on the real held-out era**, paired vs the ladder.
4. What it would yield: a **deployment/robustness** verdict (does the flexible hedge stay ahead
   out-of-sample), not a certified-optimality verdict — average metrics well-powered, tail resolvable via
   breadth.

The single-name PoC already shows the transfer holds; the wide-and-enriched version is what would turn the
simulated tail-win into a *measured* real one. The certified contribution stands on the simulator.

## Reproduce

```bash
python experiments/recover.py          # Phase 1 — frictionless recovery of BS delta
python experiments/multiseed.py        # Phase 2 — mean-variance: deep ties the correct band (boundary)
python experiments/win_test.py         # Phase 3 — tail objective: deep vs constant & WW bands, GBM + jumps
python experiments/hn_dp.py            # exact Hodges-Neuberger optimum (DP) — certifies the gap (GBM)
python experiments/robustness.py       # vol misspecification + jumps
python experiments/bootstrap_ci.py     # paired test-path bootstrap CI on the tail win (significance)
python experiments/asian_transformer.py# transformer vs FF on a path-dependent Asian option
python experiments/real_data_test.py   # proof-of-concept sim->real transfer test on SPY/AAPL
```
