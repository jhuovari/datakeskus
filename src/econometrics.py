"""Small OLS routine with Newey-West standard errors.

Only numpy is needed.  Quarterly year-on-year growth rates are serially
correlated by construction, so the HAC covariance is not optional here.
"""
from __future__ import annotations

import numpy as np


class OLS:
    def __init__(self, y: np.ndarray, X: np.ndarray, names: list[str] | None = None,
                 add_const: bool = True, hac_lags: int | None = None):
        y = np.asarray(y, float).ravel()
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        if add_const:
            X = np.column_stack([np.ones(len(X)), X])
            names = ["const"] + (names or [f"x{i}" for i in range(X.shape[1] - 1)])
        else:
            names = names or [f"x{i}" for i in range(X.shape[1])]

        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, X = y[ok], X[ok]
        self.n, self.k = X.shape
        self.names, self.y, self.X = names, y, X

        XtX_inv = np.linalg.pinv(X.T @ X)
        self.beta = XtX_inv @ X.T @ y
        self.resid = y - X @ self.beta
        self.dof = self.n - self.k
        ss_res = float(self.resid @ self.resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self.sigma2 = ss_res / self.dof

        L = hac_lags if hac_lags is not None else int(np.floor(4 * (self.n / 100) ** (2 / 9)))
        self.hac_lags = L
        u = self.resid[:, None] * X
        S = u.T @ u
        for l in range(1, L + 1):
            w = 1 - l / (L + 1)
            G = u[l:].T @ u[:-l]
            S += w * (G + G.T)
        self.vcov = XtX_inv @ S @ XtX_inv
        self.se = np.sqrt(np.diag(self.vcov))
        self.t = self.beta / self.se

    def summary(self) -> str:
        lines = [f"  n={self.n}  R2={self.r2:.3f}  HAC lags={self.hac_lags}",
                 f"  {'muuttuja':<22s}{'kerroin':>11s}{'HAC s.e.':>11s}{'t':>8s}"]
        for nm, b, s, t in zip(self.names, self.beta, self.se, self.t):
            lines.append(f"  {nm:<22s}{b:>11.4f}{s:>11.4f}{t:>8.2f}")
        return "\n".join(lines)
