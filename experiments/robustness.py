"""Robustness — the boundary discipline from the liquidation project, applied to hedging.

R1  VOL MISSPECIFICATION: all strategies calibrated at sigma=0.20; test on true sigma in
    {0.15,0.20,0.25,0.30}. Does the deep hedge degrade gracefully vs the tuned band?
R2  JUMP RISK: train the deep hedge ON Merton jump-diffusion; test on jumps. Delta cannot
    hedge jump gap risk (short-gamma fat left tail), so the incomplete-market edge should WIDEN.
Objective: mean-variance (RMSE); cost c=1%; paired CRN test.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_hedging import (
    bs_price_call, simulate_gbm, simulate_merton, Hedger, rollout_pnl,
    delta_hedge_pnl, band_hedge_pnl, train_hedger,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S0, K, SIGMA, T, N, C = 100.0, 100.0, 0.20, 1.0, 30, 0.01
premium = float(bs_price_call(torch.tensor(S0), K, T, SIGMA))
def rmse(W): W = W.double().cpu(); return float(torch.sqrt((W*W).mean()))
def cvar99(W): W=W.double().cpu(); L=-W; q=torch.quantile(L,0.99); return float(q+100*torch.clamp(L-q,min=0).mean())
H_GRID = torch.linspace(0.0, 0.40, 41)

# ---------------- R1: vol misspecification ----------------
print("="*70); print("R1  VOL MISSPECIFICATION  (all calibrated at sigma=0.20, cost=1%)"); print("="*70)
torch.manual_seed(0)
hed = Hedger(n_features=5, hidden=(64,64), use_prev=True).to(DEV)
train_hedger(hed, K=K, sigma=SIGMA, T=T, N=N, cost=C, premium=premium, S0=S0,
             objective="quadratic", use_prev=True, steps=1600, batch=16384, lr=1e-3, device=DEV, seed=0)
hed.eval()
# tune band on sigma=0.20 validation
gv = torch.Generator(device=DEV).manual_seed(999)
S_val = simulate_gbm(S0,0.0,SIGMA,T,N,100_000,DEV,gv)
best_h,best_r=0.0,1e18
for h in H_GRID.tolist():
    r=rmse(band_hedge_pnl(S_val,K,SIGMA,T,C,premium,h))
    if r<best_r: best_r,best_h=r,h
print(f"tuned band h*={best_h:.3f}\n{'true sig':>9}{'delta@.20':>11}{'band@.20':>10}{'deep@.20':>10}"
      f"{'deep vs band':>14}{'(oracle d@true)':>16}")
for sig_true in (0.15, 0.20, 0.25, 0.30):
    gt = torch.Generator(device=DEV).manual_seed(7000+int(sig_true*100))
    St = simulate_gbm(S0,0.0,sig_true,T,N,200_000,DEV,gt)
    d   = rmse(delta_hedge_pnl(St,K,SIGMA,T,C,premium))
    b   = rmse(band_hedge_pnl(St,K,SIGMA,T,C,premium,best_h))
    orc = rmse(delta_hedge_pnl(St,K,sig_true,T,C,premium))         # unattainable: knows true sigma
    with torch.no_grad():
        dp = rmse(rollout_pnl(hed,St,K,SIGMA,T,C,premium,True))
    print(f"{sig_true:>9.2f}{d:>11.3f}{b:>10.3f}{dp:>10.3f}{100*(b-dp)/b:>12.1f}%{orc:>16.3f}")

# ---------------- R2: jump risk ----------------
print("\n"+"="*70); print("R2  JUMP RISK  (Merton lam=1/yr, mean jump -5%, jump vol 10%, cost=1%)"); print("="*70)
JP = dict(lam=1.0, mj=-0.05, sj=0.10)
def merton_sim(B, gen): return simulate_merton(S0,0.0,SIGMA,T,N,B,DEV,gen,**JP)
# fair premium under the (risk-neutral, mu=0) jump model, by MC
gp = torch.Generator(device=DEV).manual_seed(321)
Sp = simulate_merton(S0,0.0,SIGMA,T,N,400_000,DEV,gp,**JP)
prem_j = float(torch.clamp(Sp[:,-1]-K,min=0).double().mean())
print(f"Merton fair premium = {prem_j:.4f}  (vs BS {premium:.4f})")
# train deep ON jumps; also keep the GBM-trained deep to show train-distribution matters
torch.manual_seed(0)
hed_j = Hedger(n_features=5, hidden=(64,64), use_prev=True).to(DEV)
train_hedger(hed_j, K=K, sigma=SIGMA, T=T, N=N, cost=C, premium=prem_j, S0=S0,
             objective="quadratic", use_prev=True, steps=1600, batch=16384, lr=1e-3,
             device=DEV, seed=0, sim=merton_sim)
hed_j.eval()
# test on jump paths
gt = torch.Generator(device=DEV).manual_seed(55555)
Sj = simulate_merton(S0,0.0,SIGMA,T,N,200_000,DEV,gt,**JP)
# tune band on jump validation
gvj = torch.Generator(device=DEV).manual_seed(444)
Svj = simulate_merton(S0,0.0,SIGMA,T,N,100_000,DEV,gvj,**JP)
best_h,best_r=0.0,1e18
for h in H_GRID.tolist():
    r=rmse(band_hedge_pnl(Svj,K,SIGMA,T,C,prem_j,h))
    if r<best_r: best_r,best_h=r,h
d_j  = delta_hedge_pnl(Sj,K,SIGMA,T,C,prem_j)
b_j  = band_hedge_pnl(Sj,K,SIGMA,T,C,prem_j,best_h)
with torch.no_grad():
    dp_j = rollout_pnl(hed_j,Sj,K,SIGMA,T,C,prem_j,True)
    dp_gbm = rollout_pnl(hed,Sj,K,SIGMA,T,C,prem_j,True)    # GBM-trained deep on jumps (mismatched); same premium for fairness
print(f"tuned band h*={best_h:.3f}")
print(f"{'strategy':<26}{'RMSE':>8}{'CVaR99':>9}   (on jump-diffusion test)")
for name,W in [("naive delta",d_j),("tuned band",b_j),
               ("deep (trained on GBM)",dp_gbm),("deep (trained on jumps)",dp_j)]:
    print(f"{name:<26}{rmse(W):>8.3f}{cvar99(W):>9.3f}")
print(f"\ndeep-on-jumps vs delta:  RMSE {100*(rmse(d_j)-rmse(dp_j))/rmse(d_j):+.1f}%   "
      f"CVaR99 {100*(cvar99(d_j)-cvar99(dp_j))/cvar99(d_j):+.1f}%")
print(f"deep-on-jumps vs band :  RMSE {100*(rmse(b_j)-rmse(dp_j))/rmse(b_j):+.1f}%   "
      f"CVaR99 {100*(cvar99(b_j)-cvar99(dp_j))/cvar99(b_j):+.1f}%")
