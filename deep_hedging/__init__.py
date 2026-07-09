"""deep_hedging — a small library in three sections:
    data       market simulators (GBM, Merton) + real-data fetch
    baselines  BS analytics + classical hedges (delta, Leland, no-trade band, Whalley-Wilmott)
    model      the neural hedger, its features, the differentiable rollout, and pathwise training
    risk       risk measures (quadratic / entropic / CVaR)
The public API is re-exported here so experiments can `from deep_hedging import ...`.
"""
from .data import simulate_gbm, simulate_merton, fetch
from .baselines import (
    bs_price_call, bs_delta_call, bs_gamma_call,
    delta_hedge_pnl, band_hedge_pnl, ww_band_hedge_pnl, leland_vol,
)
from .model import Hedger, make_features, rollout_pnl, train_hedger
from .risk import entropic_risk, cvar_loss, quadratic_risk

__all__ = [
    "simulate_gbm", "simulate_merton", "fetch",
    "bs_price_call", "bs_delta_call", "bs_gamma_call",
    "delta_hedge_pnl", "band_hedge_pnl", "ww_band_hedge_pnl", "leland_vol",
    "Hedger", "make_features", "rollout_pnl", "train_hedger",
    "entropic_risk", "cvar_loss", "quadratic_risk",
]
