"""Risk measures on terminal wealth W (higher wealth = better; these are risks to MINIMIZE)."""
import math
import torch


def entropic_risk(W, lam):
    """rho(W) = (1/lam) log E[exp(-lam W)]  (convex, decreasing in wealth; = exponential utility)."""
    return (torch.logsumexp(-lam * W, dim=0) - math.log(W.shape[0])) / lam


def cvar_loss(W, alpha):
    """CVaR_alpha of the LOSS L=-W, via Rockafellar-Uryasev (returns the risk, minimize)."""
    L = -W
    var = torch.quantile(L, alpha)
    return var + (1.0 / (1.0 - alpha)) * torch.clamp(L - var, min=0.0).mean()


def quadratic_risk(W):
    """Mean-square terminal wealth; with premium fixed at fair value this is ~ variance."""
    return (W * W).mean()
