"""Translate the Google €13 bn data-centre programme into economic shocks.

The single most consequential step in the whole analysis is *not* the
multiplier -- it is deciding how much of a euro of AI data-centre capex ever
becomes demand for Finnish production.  An AI campus is mostly a warehouse full
of imported accelerators.  This module therefore keeps the capex split and the
import assumptions explicit, parameterised and sourced, and derives the
import-penetration ratios it can from the Statistics Finland tables themselves
rather than asserting them.

Announced facts (Google press release 9.9.2026, Alphabet/Fortum releases)
------------------------------------------------------------------------
* "at least €13 billion in digital infrastructure in Finland over the next two
  years (2027-2028)"
* Sites: Hamina (expansion), Kajaani, Muhos, Vaala (new)
* 22-year PPA with Fortum tied to the Loviisa life extension
* onshore wind supported by Google raised to 629 MW (Valorem, Suomen Hyötytuuli)
* 94 MW battery near Kajaani, operational late 2027
* €31 m community funding over four years, €10 m of it for R&I
* Company's own impact claims, which this analysis tests:
  - "€3.6 billion annual average contribution to Finland's GDP" in 2027-2028
  - "more than 37,000 jobs nationwide", "approximately 16,000 ... in
    construction"
  - operations phase "7,000 jobs annually", wages 24% above median
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ----------------------------------------------------------------- headline facts
TOTAL_CAPEX = 13_000.0        # M EUR, "at least 13 bn"
YEARS = ("2027", "2028")
CAPEX_PER_YEAR = TOTAL_CAPEX / len(YEARS)

GOOGLE_CLAIMS = {
    "gdp_per_year_meur": 3_600.0,
    "jobs_total": 37_000.0,
    "jobs_construction": 16_000.0,
    "jobs_operations_per_year": 7_000.0,
    "wage_premium_vs_median": 0.24,
}

# Contracted energy side (not part of the €13 bn, but induced by the deal)
WIND_MW_NEW = 629.0
BATTERY_MW = 94.0
WIND_CAPEX_PER_MW = 1.15         # M EUR/MW, Finnish onshore wind 2025-2027
BATTERY_CAPEX_PER_MW = 0.55      # M EUR/MW, 2h grid-scale Li-ion
WIND_DOMESTIC_CONTENT = 0.35     # turbines imported; civils, roads, grid, erection domestic
BATTERY_DOMESTIC_CONTENT = 0.25

# ------------------------------------------------------- capex composition
# Level 1: IT hardware vs. the physical facility.
# Sources: PwC data-centre outlook 2026 ("ICT equipment ~70% of total capex in
# 2026"); Epoch AI TCO decomposition of a 1 GW AI campus; BloombergNEF
# hyperscaler capex split (~$240bn of $635-670bn 2026 guidance to physical
# infrastructure, i.e. ~36% -> IT ~64%).  We centre on 68% and bracket it.
IT_SHARE = {"low_import": 0.55, "central": 0.68, "high_import": 0.78}

# Level 2: composition of the *facility* (non-IT) capex.  Standard hyperscale
# fit-out engineering splits; electrical dominates because of redundancy.
FACILITY_SPLIT = {
    "electrical": 0.42,     # transformers, switchgear, UPS, gensets, busway
    "mechanical": 0.18,     # chillers, CRAH/CDU, liquid cooling, piping
    "shell_civil": 0.27,    # buildings, foundations, site works, roads
    "engineering": 0.09,    # design, project management, commissioning
    "land_other": 0.04,     # land, permits, connection fees (transfers)
}

# Level 3: for each capex element, the share that is *directly imported* and the
# Finnish industry that supplies the rest.  Import shares for equipment are
# cross-checked against measured import penetration (see `import_penetration`).
#
# element -> (import share, {industry: share of the domestic remainder})
ELEMENT_MAPPING: dict[str, tuple[float, dict[str, float]]] = {
    # Accelerator servers, storage, networking.  Finland has no volume
    # production of any of it; what stays home is logistics, wholesale margin
    # and the labour of racking, cabling and commissioning.
    "it_hardware": (0.94, {
        "F": 0.35,          # data-hall fit-out, cable trays, racking labour
        "C33": 0.20,        # installation and commissioning of equipment
        "J62_J63": 0.20,    # IT services, systems integration
        "G46": 0.10,        # wholesale margin
        "H52": 0.10,        # warehousing, freight handling
        "H49": 0.05,        # inland haulage
    }),
    "electrical": (0.70, {  # Hitachi Energy Vaasa, ABB etc. supply some of it
        "C27": 0.45,        # electrical equipment manufacturing
        "F": 0.40,          # electrical installation (part of F 43)
        "C25": 0.15,        # metal products, busbars, enclosures
    }),
    "mechanical": (0.55, {
        "C28": 0.35,        # machinery n.e.c. (chillers, pumps, air handling)
        "F": 0.50,          # HVAC and plumbing installation
        "C25": 0.15,
    }),
    "shell_civil": (0.05, {  # F's own import content is handled by the model
        "F": 1.00,
    }),
    "engineering": (0.30, {
        "M71": 0.70,        # architecture and engineering
        "M69_M70": 0.20,    # management consultancy, project management
        "J62_J63": 0.10,
    }),
    # Land purchases and permit fees are transfers, not production.  Only the
    # real-estate service margin is output.
    "land_other": (0.00, {
        "L68XL68202": 0.60,
        "O84": 0.40,
    }),
}

# Purchaser -> basic price wedge.  The IO table's own GFCF column gives the
# measured product-tax content of Finnish capital formation.
# (D21N / total at purchaser prices; computed in `price_wedges`.)


@dataclass
class Shock:
    """A final-demand shock expressed on the model's 64-industry basis."""
    name: str
    f: np.ndarray                  # domestic final demand, basic prices, M EUR
    direct_imports: float          # M EUR of directly imported goods/services
    product_taxes: float           # M EUR of taxes on products
    transfers: float               # M EUR that is not production at all
    detail: pd.DataFrame = field(repr=False, default=None)

    @property
    def total(self) -> float:
        return self.f.sum() + self.direct_imports + self.product_taxes + self.transfers

    @property
    def domestic_share(self) -> float:
        return self.f.sum() / self.total


def import_penetration(year: int = 2023) -> pd.Series:
    """Measured import share of domestic use, by product, from StatFin tables.

    For each product we compare total imported use (import use table, all using
    industries and final demand) against domestic output supplied to the home
    market (symmetric table row total less exports).  The result is a reality
    check on the assumed import shares in ELEMENT_MAPPING.
    """
    io = pd.read_csv(os.path.join(PROC, "io_table.csv"))
    io = io[io.timeperiod_y == year]
    R = "yhdistelma_1_20180101-tol_tt_lisa_rivi"
    C = "yhdistelma_1_20180101-tol_tt_lisa_sarake"
    dom_home = (io[(io[C] == "USE_PH")].set_index(R)["value"]
                - io[(io[C] == "P6K")].set_index(R)["value"])

    imp = pd.read_csv(os.path.join(PROC, "io_imports.csv"))
    imp = imp[imp.timeperiod_y == year]
    P = "yhdistelma_2_20180101"
    I = "yhdistelma_1_20180101"
    imp_tot = imp[imp[I] == "USE_PH"].set_index(P)["value"]

    # product codes in the import table are the industry codes without prefixes
    xwalk = {"C26": "26", "C27": "27", "C28": "28", "C25": "25", "C33": "33",
             "F": "41_43", "M71": "71", "M69_M70": "69_70", "J62_J63": "62_63",
             "G46": "46", "H49": "49", "H52": "52", "D35": "35"}
    rows = {}
    for ind, prod in xwalk.items():
        if prod in imp_tot.index and ind in dom_home.index:
            m = imp_tot[prod]
            d = max(dom_home[ind], 0.0)
            rows[ind] = m / (m + d) if (m + d) > 0 else np.nan
    return pd.Series(rows, name="import_penetration")


def price_wedges(year: int = 2023) -> dict:
    """Product-tax and import content of Finnish gross fixed capital formation."""
    io = pd.read_csv(os.path.join(PROC, "io_table.csv"))
    io = io[io.timeperiod_y == year]
    R = "yhdistelma_1_20180101-tol_tt_lisa_rivi"
    C = "yhdistelma_1_20180101-tol_tt_lisa_sarake"
    col = io[io[C] == "P51K"].set_index(R)["value"]
    total = col["FIMUSE_OH"]
    return {
        "gfcf_total_purchaser": float(total),
        "gfcf_domestic_basic": float(col["P1_USE"]),
        "gfcf_imports": float(col["P7_USE"]),
        "gfcf_product_taxes": float(col["D21N"]),
        "product_tax_rate": float(col["D21N"] / total),
        "import_share": float(col["P7_USE"] / total),
        "domestic_share": float(col["P1_USE"] / total),
    }


def build_shock(industries: list[str], capex: float = TOTAL_CAPEX,
                it_share_key: str = "central",
                name: str | None = None) -> Shock:
    """Map `capex` (M EUR, purchaser prices) onto a 64-industry demand vector."""
    it_share = IT_SHARE[it_share_key]
    elements = {"it_hardware": capex * it_share}
    facility = capex * (1 - it_share)
    for k, s in FACILITY_SPLIT.items():
        elements[k] = facility * s

    f = pd.Series(0.0, index=industries)
    imports = 0.0
    transfers = 0.0
    rows = []
    for el, amount in elements.items():
        imp_share, mix = ELEMENT_MAPPING[el]
        assert abs(sum(mix.values()) - 1.0) < 1e-9, f"{el} mix must sum to 1"
        imported = amount * imp_share
        domestic = amount - imported
        if el == "land_other":
            # 70% of this line is pure land/permit transfer, only the rest is a
            # real-estate and administrative service margin.
            transfers += domestic * 0.70
            domestic *= 0.30
        imports += imported
        for ind, w in mix.items():
            f[ind] += domestic * w
        rows.append({"element": el, "amount": amount, "imported": imported,
                     "domestic_demand": domestic,
                     "import_share": imp_share})

    # Strip product taxes out of the domestic demand so the vector is at basic
    # prices, consistent with the IO table.
    w = price_wedges()
    tax_rate = w["product_tax_rate"]
    product_taxes = f.sum() * tax_rate
    f = f * (1 - tax_rate)

    return Shock(
        name=name or f"capex {capex:,.0f} M EUR, IT share {it_share:.0%}",
        f=f.to_numpy(),
        direct_imports=imports,
        product_taxes=product_taxes,
        transfers=transfers,
        detail=pd.DataFrame(rows),
    )


def energy_shock(industries: list[str]) -> Shock:
    """Wind and battery capex contracted alongside the data-centre programme.

    This sits *outside* the €13 bn: Google buys the power, developers build the
    plant.  It is nonetheless investment in Finland that the deal triggers.
    """
    wind = WIND_MW_NEW * WIND_CAPEX_PER_MW
    batt = BATTERY_MW * BATTERY_CAPEX_PER_MW
    f = pd.Series(0.0, index=industries)
    imports = 0.0
    for amount, dom in ((wind, WIND_DOMESTIC_CONTENT), (batt, BATTERY_DOMESTIC_CONTENT)):
        imports += amount * (1 - dom)
        d = amount * dom
        # Domestic content of a Finnish wind/storage project: civil works and
        # grid connection (F), erection and electrical (C27/F), engineering.
        f["F"] += d * 0.60
        f["C27"] += d * 0.15
        f["M71"] += d * 0.15
        f["H49"] += d * 0.10
    tax_rate = price_wedges()["product_tax_rate"]
    taxes = f.sum() * tax_rate
    return Shock(name=f"energy capex {wind + batt:,.0f} M EUR",
                 f=(f * (1 - tax_rate)).to_numpy(),
                 direct_imports=imports, product_taxes=taxes, transfers=0.0)
