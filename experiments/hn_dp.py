"""The decisive test: exact Hodges-Neuberger optimum by dynamic programming.

The entropic risk rho_lambda(W) = (1/lambda) log E[exp(-lambda W)] is exponential utility, whose
EXACT cost-aware optimal hedge under proportional cost is the Hodges-Neuberger (1989) indifference
hedge -- a state-dependent no-trade region. Whalley-Wilmott is only its leading-order asymptotic.
We compute the EXACT optimum by DP on (log-spot, holding) and compare it to the deep hedge and the
classical bands. This mirrors the liquidation project's exact-optimum oracle: if deep ~ HN, the deep
hedge MATCHES the true optimum (a boundary, beating only the asymptotic/heuristic bands); if deep is
still well above HN, there is room the deep hedge is not capturing.

Log-space DP (avoids exp overflow):
  L_N(logS,h)      = lambda*( c*S*|h| + (S-K)^+ )                          [liquidate + pay payoff]
  E(logS,h_i)      = logsumexp_z[ logw_z - lambda*h_i*(S'_z - S) + L_{i+1}(logS'_z, h_i) ]
  L_i(logS,h_prev) = min_{h_i} [ lambda*c*S*|h_i-h_prev| + E(logS,h_i) ]
  rho_HN           = L_0(logS0, h=0)/lambda - premium
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import bs_price_call

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
S0, K, SIGMA, T, N, LAM = 100.0, 100.0, 0.20, 1.0, 30, 1.0
dt = T / N

# grids (uniform log-S for O(1) interpolation)
nS, nH, nQ = 601, 301, 25
logS = torch.linspace(math.log(20.0), math.log(500.0), nS, device=DEV, dtype=DT)
dlog = (logS[1] - logS[0]).item()
Sg = torch.exp(logS)
hgrid = torch.linspace(-0.15, 1.15, nH, device=DEV, dtype=DT)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(nQ)          # E[f(Z)], Z~N(0,1): sum w f(x)/sqrt(2pi)... use probabilists'
gh_x = torch.tensor(gh_x, device=DEV, dtype=DT); gh_w = torch.tensor(gh_w, device=DEV, dtype=DT)
logw = torch.log(gh_w / math.sqrt(2*math.pi))               # weights for E[f(Z)] = sum (w/sqrt(2pi)) f(x)

def interp_idx(x):
    """linear-interp index+frac on the uniform logS grid for query points x (any shape)."""
    t = (x - logS[0]) / dlog
    i0 = torch.clamp(torch.floor(t).long(), 0, nS - 2)
    return i0, (t - i0.to(DT)).clamp(0.0, 1.0)

def solve_hn(cost, jump=None):
    drift = (-0.5 * SIGMA**2) * dt
    S = Sg.unsqueeze(1)                                      # [nS,1]
    # terminal
    L = LAM * (cost * Sg[:, None] * hgrid.abs()[None, :] + torch.clamp(Sg - K, min=0.0)[:, None])  # [nS,nH]
    # precompute diffusion successor log-spots [nS,nQ] (+ optional jumps handled by mixture below)
    for i in range(N - 1, -1, -1):
        # successor log-spot for the diffusion part
        lsp = logS[:, None] + drift + SIGMA * math.sqrt(dt) * gh_x[None, :]      # [nS,nQ]
        if jump is None:
            comps = [(0.0, lsp, logw[None, :].expand(nS, nQ))]                   # (unused, lsp, logweight)
        else:
            lam_j, mj, sj = jump['lam'], jump['mj'], jump['sj']
            k_comp = math.exp(mj + 0.5*sj*sj) - 1.0
            comps = []
            for nj in range(0, 5):          # Poisson-weighted # of jumps; diffusion + nj jumps = ONE Gaussian
                logpois = nj*math.log(lam_j*dt) - lam_j*dt - math.lgamma(nj+1)
                mean = (-0.5*SIGMA**2)*dt + nj*mj - lam_j*k_comp*dt
                std = math.sqrt(SIGMA*SIGMA*dt + nj*sj*sj)
                lsp_j = logS[:, None] + mean + std*gh_x[None, :]
                comps.append((0.0, lsp_j, (logw + logpois)[None, :].expand(nS, nQ)))
        Sp = Sg[:, None]                                                        # current S [nS,1]
        # E(logS, h_i) via logsumexp over quadrature (and jump comps)
        E = torch.full((nS, nH), -float('inf'), device=DEV, dtype=DT)
        stack_terms = []
        for (_, lsp_c, lw_c) in comps:
            i0, fr = interp_idx(lsp_c)                                          # [nS,nQ]
            Sp_c = torch.exp(lsp_c)                                             # [nS,nQ]
            # for each h_i: term_z = lw_c - lam*h_i*(Sp_c - Sp) + L_interp(:,:,h_i)
            L0 = L[i0]                                                          # [nS,nQ,nH]
            L1 = L[i0 + 1]
            Lint = (1 - fr).unsqueeze(-1) * L0 + fr.unsqueeze(-1) * L1          # [nS,nQ,nH]
            payoff_move = -LAM * hgrid[None, None, :] * (Sp_c[..., None] - Sp[..., None])  # [nS,nQ,nH]
            term = lw_c[..., None] + payoff_move + Lint                         # [nS,nQ,nH]
            stack_terms.append(term)
        allterm = torch.cat(stack_terms, dim=1)                                # [nS, nQ*ncomp, nH]
        E = torch.logsumexp(allterm, dim=1)                                    # [nS,nH]
        # L_i(logS,h_prev) = min_{h_i} lam*c*S*|h_i-h_prev| + E
        tradecost = LAM * cost * Sg[:, None, None] * (hgrid[None, :, None] - hgrid[None, None, :]).abs()  # [nS,nHprev,nHi]
        L = (tradecost + E[:, None, :]).min(dim=2).values                      # [nS,nHprev]
    # rho at (S0, h_prev=0)
    iS0 = int(torch.argmin((logS - math.log(S0)).abs()).item())
    ih0 = int(torch.argmin(hgrid.abs()).item())
    return L[iS0, ih0].item()

prem = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
for cost in (0.0, 0.01):
    rho_hn = solve_hn(cost) / LAM - prem
    print(f"GBM     c={cost:.3f}:  rho_HN (exact entropic optimum) = {rho_hn:.4f}   (premium {prem:.3f})")
print("  compare GBM c=.01: deep 3.510 | const band 4.508 | WW band 5.139  -> HN is the floor\n")

# Merton (jumps): exact optimum with the jump transition; fair premium by MC on risk-neutral jumps
from deep_hedging import simulate_merton
JP = dict(lam=1.0, mj=-0.05, sj=0.10)
gp = torch.Generator(device=DEV).manual_seed(321)
prem_j = float(torch.clamp(simulate_merton(S0,0.0,SIGMA,T,N,400_000,DEV,gp,**JP)[:, -1]-K, min=0).double().mean())
rho_hn_j = solve_hn(0.01, jump=JP) / LAM - prem_j
print(f"Merton  c=0.010:  rho_HN = {rho_hn_j:.4f}   (premium {prem_j:.3f})")
print("  compare Merton c=.01: deep 14.276 | const band 23.985 | WW band 24.889  -> HN is the floor")
