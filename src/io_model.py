"""Leontief input-output model of the Finnish economy, built from StatFin data.

Data source: Statistics Finland, symmetric input-output table at basic prices
(table pt/14yn), import use table (pt/14yp) and labour input (pt/14yr).

The module exposes

    IOSystem.load(year)      -> assembled system, with A, L, and coefficients
    sys.impact(f)            -> Type I impact of a final-demand vector f
    sys.impact(f, type2=True)-> Type II impact (income-induced consumption on top)

Conventions
-----------
* All monetary values are millions of euro at basic prices, current prices of
  the table year.  Final-demand shocks handed to `impact` must therefore be
  expressed at basic prices too (purchaser-price shocks must first have trade
  and transport margins and product taxes stripped out -- `scenario.py` does
  that explicitly).
* Row/column order is the 64 market industries of the table, in table order.
* `x` is output at basic prices, `A = Z x_hat^-1` the domestic technical
  coefficient matrix, `L = (I - A)^-1` the Leontief inverse.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ROW = "yhdistelma_1_20180101-tol_tt_lisa_rivi"
COL = "yhdistelma_1_20180101-tol_tt_lisa_sarake"

# Rows of pt/14yn that are *not* supplying industries but accounting items.
AGGREGATE_ROWS = {
    "P1_USE", "P7_USE", "PUR_S2", "PUR_F", "CIF_FOB", "D21N", "FIMUSE_OH",
    "D1K", "D29N", "P51CK", "B13N", "B1GPH", "P1R", "P51K", "NKANTA", "E1",
}
# Columns of pt/14yn that are final demand or totals rather than industries.
AGGREGATE_COLS = {
    "ATX", "P3KS14", "P3KS15", "P3KS13", "P3K", "P51K", "P52K", "P5K",
    "P6KS21", "P6KS2111", "P6KS2112", "P6KS22", "P6K", "FUSE_PH", "USE_PH",
}


@dataclass
class IOSystem:
    year: int
    industries: list[str]
    labels: dict[str, str]
    Z: np.ndarray            # 64x64 domestic intermediate flows
    x: np.ndarray            # output at basic prices
    A: np.ndarray            # technical coefficients (domestic)
    L: np.ndarray            # Leontief inverse
    va: np.ndarray           # gross value added at basic prices
    comp: np.ndarray         # compensation of employees
    gos: np.ndarray          # net operating surplus / mixed income
    cfc: np.ndarray          # consumption of fixed capital
    othertax: np.ndarray     # other taxes less subsidies on production
    emp: np.ndarray          # employed persons, 1000
    imp_int: np.ndarray      # imported intermediates used, by using industry
    final_demand: pd.DataFrame   # domestic-product final demand columns
    fd_imports: pd.Series        # imported product content of final demand cols
    household_params: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def n(self) -> int:
        return len(self.industries)

    @property
    def v(self) -> np.ndarray:
        """Value added per unit of output."""
        return _safe_div(self.va, self.x)

    @property
    def wage_coef(self) -> np.ndarray:
        return _safe_div(self.comp, self.x)

    @property
    def emp_coef(self) -> np.ndarray:
        """Employed persons per million euro of output."""
        return _safe_div(self.emp * 1000.0, self.x)

    @property
    def imp_coef(self) -> np.ndarray:
        """Imported intermediates per unit of output."""
        return _safe_div(self.imp_int, self.x)

    def idx(self, code: str) -> int:
        return self.industries.index(code)

    # ------------------------------------------------------------ multipliers
    def output_multiplier(self) -> np.ndarray:
        """Type I output multiplier: column sums of L."""
        return self.L.sum(axis=0)

    def va_multiplier(self) -> np.ndarray:
        """Value added generated per unit of final demand (Type I)."""
        return self.v @ self.L

    def emp_multiplier(self) -> np.ndarray:
        """Persons per M EUR of final demand (Type I)."""
        return self.emp_coef @ self.L

    # -------------------------------------------------------- Type II closure
    def closed_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Augment A with a household row and a household consumption column.

        Household row  h_j  = household disposable income generated per unit of
        output in industry j.  We take compensation of employees, strip the part
        that never reaches households as spendable income (employers' social
        contributions, employees' contributions and direct taxes), and keep the
        rest.

        Household column c_i = share of an extra euro of disposable income spent
        on the *domestic* output of industry i (i.e. average propensity to
        consume x domestic content x product-i share of consumption).

        Returns (A_closed, c_dom) where A_closed is (n+1)x(n+1).
        """
        p = self.household_params
        # Compensation of employees -> household spendable income.
        net_of_wedge = p["wage_to_net_income"]
        h = self.wage_coef * net_of_wedge

        # Domestic consumption pattern per euro of disposable income.
        cons = self.final_demand["P3KS14"].reindex(self.industries).fillna(0.0).to_numpy()
        c_dom = cons / cons.sum() * p["apc_domestic"]

        n = self.n
        Ac = np.zeros((n + 1, n + 1))
        Ac[:n, :n] = self.A
        Ac[:n, n] = c_dom          # households buy domestic output
        Ac[n, :n] = h              # industries pay income to households
        return Ac, c_dom

    # ---------------------------------------------------------------- impacts
    def impact(self, f: np.ndarray, type2: bool = False) -> dict:
        """Total requirements from a final-demand vector `f` (basic prices).

        Returns a dict of aggregate impacts plus the industry-level output
        vector, so callers can report the sectoral pattern.
        """
        f = np.asarray(f, dtype=float)
        assert f.shape == (self.n,), f"expected shape ({self.n},), got {f.shape}"

        if not type2:
            x_ind = self.L @ f
            induced = 0.0
        else:
            Ac, _ = self.closed_system()
            fc = np.append(f, 0.0)
            Lc = np.linalg.inv(np.eye(self.n + 1) - Ac)
            xc = Lc @ fc
            x_ind = xc[: self.n]
            induced = xc[self.n]

        return {
            "output": x_ind,
            "output_total": x_ind.sum(),
            "va_total": float(self.v @ x_ind),
            "comp_total": float(self.wage_coef @ x_ind),
            "gos_total": float(_safe_div(self.gos, self.x) @ x_ind),
            "cfc_total": float(_safe_div(self.cfc, self.x) @ x_ind),
            "othertax_total": float(_safe_div(self.othertax, self.x) @ x_ind),
            "emp_total": float(self.emp_coef @ x_ind),          # persons
            "imports_induced": float(self.imp_coef @ x_ind),    # intermediate imports
            "household_income": float(induced),
        }

    # ------------------------------------------------------------------- load
    @classmethod
    def load(cls, year: int = 2023) -> "IOSystem":
        io = pd.read_csv(os.path.join(PROC, "io_table.csv"))
        io = io[io.timeperiod_y == year]
        labels = dict(zip(io[ROW], io["label_" + ROW]))
        labels.update(dict(zip(io[COL], io["label_" + COL])))

        row_order = list(dict.fromkeys(io[ROW]))
        col_order = list(dict.fromkeys(io[COL]))
        inds = [c for c in col_order if c not in AGGREGATE_COLS]
        assert inds == [r for r in row_order if r not in AGGREGATE_ROWS], \
            "row and column industry orders must match"

        wide = io.pivot_table(index=ROW, columns=COL, values="value", aggfunc="sum")
        wide = wide.reindex(index=row_order, columns=col_order).fillna(0.0)

        Z = wide.loc[inds, inds].to_numpy(dtype=float)
        x = wide.loc["P1R", inds].to_numpy(dtype=float)
        va = wide.loc["B1GPH", inds].to_numpy(dtype=float)
        comp = wide.loc["D1K", inds].to_numpy(dtype=float)
        gos = wide.loc["B13N", inds].to_numpy(dtype=float)
        cfc = wide.loc["P51CK", inds].to_numpy(dtype=float)
        othertax = wide.loc["D29N", inds].to_numpy(dtype=float)
        emp = wide.loc["E1", inds].to_numpy(dtype=float)
        imp_int = wide.loc["P7_USE", inds].to_numpy(dtype=float)

        A = Z / np.where(x == 0, np.nan, x)          # column-normalise by output
        A = np.nan_to_num(A)
        L = np.linalg.inv(np.eye(len(inds)) - A)

        fd_cols = ["P3KS14", "P3KS15", "P3KS13", "P51K", "P52K", "P6K"]
        final_demand = wide.loc[inds, fd_cols]
        fd_imports = wide.loc["P7_USE", fd_cols]

        return cls(
            year=year, industries=inds, labels=labels, Z=Z, x=x, A=A, L=L,
            va=va, comp=comp, gos=gos, cfc=cfc, othertax=othertax, emp=emp,
            imp_int=imp_int, final_demand=final_demand, fd_imports=fd_imports,
            household_params=household_calibration(year),
        )


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, float) / np.where(np.asarray(b, float) == 0, np.nan, b))


def household_calibration(year: int) -> dict:
    """Derive the Type II closure parameters from the household sector account.

    wage_to_net_income : share of employers' total labour cost (D1) that ends up
        as household income net of direct taxes and social contributions.
    apc_domestic : share of an extra euro of disposable income spent on
        *domestically produced* consumption goods and services.
    """
    hh = pd.read_csv(os.path.join(PROC, "na_household.csv"))
    hh = hh[(hh.sektoriluokitus_7_20230101 == "S14") & (hh.contentscode == "ntp-cp")]
    s = hh[hh.timeperiod_y == year].set_index("taloustoimi_1_20180101")["value"]

    d1 = s["D1R"]              # compensation received (incl. employers' contributions)
    d12 = s["D12R"]            # employers' social contributions -> never spendable
    d5k = s["D5K"]             # direct taxes paid by households
    d613 = s["D613K"]          # households' own social contributions
    b6g = s["B6G"]             # gross disposable income
    p3k = s["P3K"]             # consumption expenditure

    # Direct tax + own-contribution wedge, measured against total primary income
    # actually accruing to households (proxy: disposable income + the wedge).
    wedge_rate = (d5k + d613) / (b6g + d5k + d613)
    wage_to_net = (d1 - d12) / d1 * (1 - wedge_rate)

    apc = p3k / b6g            # average propensity to consume out of disposable income

    # Domestic content of household consumption, from the IO table itself.
    io = pd.read_csv(os.path.join(PROC, "io_table.csv"))
    io = io[io.timeperiod_y == year]
    cons = io[io[COL] == "P3KS14"].set_index(ROW)["value"]
    dom = cons["P1_USE"]
    imp = cons["P7_USE"] + cons.get("PUR_S2", 0.0)
    dom_share = dom / (dom + imp)

    return {
        "d1": float(d1), "employer_contrib_share": float(d12 / d1),
        "direct_tax_wedge": float(wedge_rate),
        "wage_to_net_income": float(wage_to_net),
        "apc": float(apc),
        "consumption_domestic_share": float(dom_share),
        "apc_domestic": float(apc * dom_share),
    }
