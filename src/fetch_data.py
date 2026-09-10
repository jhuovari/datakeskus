"""Fetch every dataset the analysis needs from Statistics Finland and cache it.

Run:  python3 src/fetch_data.py
Everything lands in data/raw as json-stat2 and in data/processed as tidy CSV.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import statfin  # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ---------------------------------------------------------------- IO framework
IO_YEAR = "2023"          # latest symmetric input-output table
NA_YEARS = [str(y) for y in range(1995, 2026)]

DATASETS = {
    # Symmetric industry x industry input-output table, basic prices.
    # Rows = supplying industries + value-added/employment rows,
    # columns = using industries + final demand categories.
    "io_table": ("pt/14yn.px", {
        "timeperiod_y": [IO_YEAR],
        "yhdistelma_1_20180101-tol_tt_lisa_rivi": ["*"],
        "yhdistelma_1_20180101-tol_tt_lisa_sarake": ["*"],
        "contentscode": ["pt-cp"],
    }),
    # Use of *imported* products, basic prices -> gives us import leakage
    # per using industry and per final-demand category.
    "io_imports": ("pt/14yp.px", {
        "timeperiod_y": [IO_YEAR],
        "yhdistelma_2_20180101": ["*"],
        "yhdistelma_1_20180101": ["*"],
        "contentscode": ["pt-cp"],
    }),
    # Labour input per unit of output, direct and total.
    "io_labour": ("pt/14yr.px", {
        "timeperiod_y": [IO_YEAR],
        "toimiala_79_20180101": ["*"],
        "taloustoimi_1_20180101": ["*"],
        "contentscode": ["pt-cp"],
    }),
    # Cost components of output at basic prices (compensation, taxes, GOS).
    "io_costs": ("pt/14ys.px", {
        "timeperiod_y": [IO_YEAR],
        "contentscode": ["*"],
    }),
    # GDP, supply and demand, annual, current prices + volume changes.
    "na_gdp": ("ntp/15a9.px", {
        "taloustoimi_1_20180101": ["*"],
        "timeperiod_y": NA_YEARS,
        "contentscode": ["ntp-cp", "ntp-vol_muutos"],
    }),
    # Employment and hours by industry, whole economy.  The full cube exceeds
    # PxWeb's cell limit, so we pin the sector to S1 and take only persons/hours.
    "na_employment": ("ntp/15ab.px", {
        "taloustoimi_1_20180101": ["E1", "E2"],
        "sektoriluokitus_7_20230101": ["S1"],
        "toimiala_79_20180101": ["*"],
        "timeperiod_y": NA_YEARS,
        "contentscode": ["*"],
    }),
    # Taxes and tax-like charges, annual (for the fiscal block).
    "na_taxes": ("ntp/15aj.px", {
        "sektoriluokitus_7_20230101": ["S13", "S1311", "S1313"],
        "verolaji_11_20190101": ["*"],
        "timeperiod_y": NA_YEARS,
        "contentscode": ["ntp-cp"],
    }),
    # Fixed capital formation and capital stock by asset type and industry --
    # used to place the data-centre capex inside Finland's investment aggregate.
    "na_capital": ("ntp/15af.px", {
        "taloustoimi_1_20180101": ["P51K", "P51CK", "NKANTA", "BKANTA"],
        "sektoriluokitus_7_20230101": ["S1"],
        "toimiala_79_20180101": ["SSS"],
        "varojenluokitus_5_20180101": ["*"],
        "timeperiod_y": NA_YEARS,
        "contentscode": ["*"],
    }),
    # Household-sector income account -- calibrates the Type II (income-induced)
    # closure: wage share going to households, direct tax + contribution wedge,
    # and the average propensity to consume.
    "na_household": ("ntp/15ac.px", {
        "sektoriluokitus_7_20230101": ["S14", "S1"],
        "taloustoimi_1_20180101": ["D1R", "D11R", "D12R", "D1K", "D11K", "D12K",
                                   "D5K", "D613K", "D62R", "B6G", "P3K", "B8G"],
        "timeperiod_y": NA_YEARS,
        "contentscode": ["ntp-cp"],
    }),
    # Labour force survey: unemployment, employment and participation rates.
    "lfs_annual": ("tyti/13aj.px", {
        "sukupuoli_9_20180101": ["SSS"],
        "ikaryhma_19_20190101": ["15-74", "15-64"],
        "timeperiod_y": ["*"],
        "contentscode": ["*"],
    }),
    # Employment and hours by industry, quarterly -- construction cycle.
    "lfs_industry_q": ("tyti/137l.px", {"timeperiod_q": ["*"], "contentscode": ["*"]}),
    # Index of wage and salary earnings by industry, quarterly -- used to
    # estimate how construction wages respond to construction activity.
    "earnings_q": ("ati/14up.px", {
        "palk_muo_2_20120101": ["0"],
        "toimiala_109_20150101": ["*"],
        "timeperiod_q": ["*"],
        "contentscode": ["ati_2015_100", "ati_vuosimuutosprosentti",
                         "spi_2015_100", "spi_vuosimuutosprosentti"],
    }),
    # Whole-economy earnings index (the by-industry table has no total).
    "earnings_total_q": ("ati/14um.px", {
        "tyonantajasekt_3_20190528": ["SSS"],
        "palk_muo_2_20120101": ["0"],
        "sukupuoli_9_20180101": ["SSS"],
        "timeperiod_q": ["*"],
        "contentscode": ["ati_2015_100", "spi_2015_100",
                         "ati_vuosimuutosprosentti", "spi_vuosimuutosprosentti"],
    }),
    # Monthly earnings of full-time employees -- needed to price the
    # "24% above median wage" claim for the operations phase.
    "earnings_level": ("ati/14uw.px", {
        "tyonantajasekt_3_20190528": ["SSS"], "sukupuoli_9_20180101": ["SSS"],
        "timeperiod_y": ["*"], "contentscode": ["*"],
    }),
    # Structure of earnings by industry, incl. median and deciles.
    "earnings_structure": ("pra/15b1.px", {"contentscode": ["*"]}),
    # Building cost index, long monthly series.
    "construction_costs": ("rki/13g8.px", {"timeperiod_m": ["*"], "contentscode": ["*"]}),
    # Regional labour markets -- the four host municipalities sit in Kymenlaakso,
    # Kainuu and North Ostrobothnia, where local labour supply is thin.
    "lfs_region": ("tyti/13al.px", {
        "timeperiod_y": ["*"], "alue_23_20180101": ["*"], "contentscode": ["*"],
    }),
    # Electricity prices by consumer type -- the base for the price-impact
    # calculation in the electricity-market block.
    "electricity_prices": ("ehi/13rb.px", {"timeperiod_m": ["*"], "contentscode": ["*"]}),
    # Electricity supply and total consumption, annual 1960-2024.
    "electricity": ("ehk/12sv.px", {"timeperiod_y": ["*"], "contentscode": ["*"]}),
    # Electricity consumption by sector.
    "electricity_sector": ("ehk/12vm.px", {"timeperiod_y": ["*"], "contentscode": ["*"]}),
}


def dump(name: str, table: str, sel: dict, force: bool = False) -> None:
    if force:
        # Cache keys are fixed per dataset, so a changed selection would
        # otherwise silently re-serve the old response.
        for key in (f"data_{name}", "meta_" + table.replace("/", "_").replace(".px", "")):
            path = os.path.join(statfin.CACHE_DIR, key + ".json")
            if os.path.exists(path):
                os.remove(path)
    meta = statfin.metadata(table)
    codes = {v["code"] for v in meta["variables"]}
    sel = {k: v for k, v in sel.items() if k in codes}
    # Any dimension we did not mention: take everything.
    for v in meta["variables"]:
        sel.setdefault(v["code"], ["*"])
    js = statfin.query(table, sel, cache_key="data_" + name)
    df = statfin.to_frame(js)
    os.makedirs(PROC, exist_ok=True)
    out = os.path.join(PROC, name + ".csv")
    df.to_csv(out, index=False)
    print(f"{name:22s} {table:14s} rows={len(df):>8,}  -> {os.path.relpath(out)}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, (table, sel) in DATASETS.items():
        if only and name not in only:
            continue
        try:
            dump(name, table, sel, force=force)
        except Exception as e:  # keep going; report at the end
            print(f"{name:22s} FAILED: {type(e).__name__}: {e}")
