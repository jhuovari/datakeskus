"""Synthesis -- put every approach on the same axis and reconcile them.

The five blocks answer different questions, so their headline numbers are not
substitutes.  This module lines them up explicitly:

  * construction phase (2027-2028), a temporary demand shock;
  * operating phase (2029 ->), a permanent change in the capital stock and in
    the composition of GDP;
  * and, for each, the difference between what shows up in GDP and what shows
    up in Finnish residents' income.

It also produces the single comparison the public debate needs: the announced
figures, the industry's figures, and what the official statistics imply.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from io_model import IOSystem     # noqa: E402
import scenario as sc             # noqa: E402
import m1_io_analysis as m1       # noqa: E402
import m2_keynesian as m2         # noqa: E402
import m3_growth as m3            # noqa: E402
import m4_electricity as m4       # noqa: E402
import m5_fiscal as m5            # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
GDP_2025 = m1.GDP_2025


def headline() -> pd.DataFrame:
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    t1 = s.impact(shock.f)
    t2 = s.impact(shock.f, type2=True)
    ops = m3.operations()
    casc, det = m3.gdp_gni_cascade(ops)
    pick = lambda pre: casc.loc[casc["Erä"].str.startswith(pre), "M€/vuosi"].iloc[0]  # noqa: E731
    ga = m3.growth_accounting()
    energy = sc.energy_shock(s.industries)
    e2 = s.impact(energy.f, type2=True)

    rows = [
        # --- construction phase, 2027-2028
        ("Rakennusvaihe", "Ilmoitettu investointi", sc.TOTAL_CAPEX,
         sc.TOTAL_CAPEX / m1.N_YEARS, sc.TOTAL_CAPEX / m1.N_YEARS / GDP_2025 * 100),
        ("Rakennusvaihe", "Kotimainen loppukysyntä", shock.f.sum(),
         shock.f.sum() / m1.N_YEARS, shock.f.sum() / m1.N_YEARS / GDP_2025 * 100),
        ("Rakennusvaihe", "Arvonlisäys, Type I", t1["va_total"],
         t1["va_total"] / m1.N_YEARS, t1["va_total"] / m1.N_YEARS / GDP_2025 * 100),
        ("Rakennusvaihe", "Arvonlisäys, Type II", t2["va_total"],
         t2["va_total"] / m1.N_YEARS, t2["va_total"] / m1.N_YEARS / GDP_2025 * 100),
        ("Rakennusvaihe", "Tuulivoima- ja akkuinvestoinnit (erillinen)",
         e2["va_total"], e2["va_total"] / m1.N_YEARS,
         e2["va_total"] / m1.N_YEARS / GDP_2025 * 100),
        # --- operating phase, steady state
        ("Käyttövaihe", "BKT-vaikutus", np.nan, pick("= BKT"),
         pick("= BKT") / GDP_2025 * 100),
        ("Käyttövaihe", "BKTL-vaikutus", np.nan, pick("= BKTL"),
         pick("= BKTL") / GDP_2025 * 100),
        ("Käyttövaihe", "Nettokansantulo-vaikutus", np.nan, pick("= Nettokansantulo"),
         pick("= Nettokansantulo") / GDP_2025 * 100),
        ("Käyttövaihe", "Potentiaalinen tuotanto (kasvulaskenta)", np.nan,
         ga["  vastaa BKT:ssa, M€"], ga["Potentiaalisen tuotannon nousu, % (α korj.)"]),
    ]
    df = pd.DataFrame(rows, columns=["Vaihe", "Suure", "M€ yhteensä",
                                     "M€ / vuosi", "% BKT:sta / vuosi"])
    return df


def employment_summary() -> pd.DataFrame:
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    t1 = s.impact(shock.f)
    t2 = s.impact(shock.f, type2=True)
    ops = m3.operations()
    _, det = m3.gdp_gni_cascade(ops)

    rows = [
        ("Rakennusvaihe, välitön", t1["emp_total"] - (s.emp_coef @ (s.L @ shock.f - shock.f)),
         "htv koko ohjelmalta"),
        ("Rakennusvaihe, Type I (välitön + välillinen)", t1["emp_total"],
         "htv koko ohjelmalta"),
        ("Rakennusvaihe, Type II (+ tulokerrannais)", t2["emp_total"],
         "htv koko ohjelmalta"),
        ("Rakennusvaihe, keskimäärin vuodessa", t2["emp_total"] / m1.N_YEARS,
         "henkeä 2027 ja 2028"),
        ("Käyttövaihe, datakeskusten oma henkilöstö", ops["Henkilöstö, henkeä"],
         "henkeä, pysyvä"),
        ("Käyttövaihe, hankintaketju (Type II)", det["Hankintaketjun työllisyys, htv/vuosi"],
         "htv/vuosi, pysyvä"),
        ("Käyttövaihe, yhteensä", det["Työllisyys yhteensä, henkeä"],
         "henkeä, pysyvä"),
    ]
    return pd.DataFrame(rows, columns=["Erä", "Määrä", "Yksikkö"])


def claims_table() -> pd.DataFrame:
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    t2 = s.impact(shock.f, type2=True)
    ops = m3.operations()
    casc, det = m3.gdp_gni_cascade(ops)
    pick = lambda pre: casc.loc[casc["Erä"].str.startswith(pre), "M€/vuosi"].iloc[0]  # noqa: E731
    c = m5.construction_phase()
    o = m5.operations_phase()

    rows = [
        ("BKT-vaikutus rakennusvaiheessa, M€/vuosi",
         "Google", 3_600.0, t2["va_total"] / m1.N_YEARS),
        ("Työllisyys rakennusvaiheessa",
         "Google", 37_000.0, t2["emp_total"]),
        ("  josta rakentamisessa",
         "Google", 16_000.0, s.emp_coef[s.idx("F")] * m1.type2_output(s, shock.f)[s.idx("F")]),
        ("Työllisyys käyttövaiheessa, henkeä/vuosi",
         "Google", 7_000.0, det["Työllisyys yhteensä, henkeä"]),
        ("Rakennusvaiheen verotuotto, M€",
         "FDCA/EK (skaalattu)", 1_841.7, c.loc[c["Erä"] == "YHTEENSÄ", "M€ (koko ohjelma)"].iloc[0]),
        ("Käyttövaiheen verotuotto, M€/vuosi",
         "FDCA/EK (skaalattu)", 433.3, o.loc[o["Erä"] == "YHTEENSÄ", "M€ / vuosi"].iloc[0]),
    ]
    df = pd.DataFrame(rows, columns=["Suure", "Lähde", "Julkinen väite", "Tämä analyysi"])
    df["Suhde"] = df["Tämä analyysi"] / df["Julkinen väite"]
    return df


def range_table() -> pd.DataFrame:
    """The honest answer: a range, with the assumption that drives it named."""
    s = IOSystem.load(2023)
    rows = []
    for key in ("low_import", "central", "high_import"):
        shock = sc.build_shock(s.industries, it_share_key=key)
        t2 = s.impact(shock.f, type2=True)
        for d in (0.0, 0.30):
            rows.append({
                "IT-laitteiden osuus": f"{sc.IT_SHARE[key]:.0%}",
                "Syrjäytymisaste": d,
                "Kotimaisuusaste, %": 100 * shock.domestic_share,
                "Arvonlisäys, M€/vuosi": t2["va_total"] * (1 - d) / m1.N_YEARS,
                "% BKT:sta": t2["va_total"] * (1 - d) / m1.N_YEARS / GDP_2025 * 100,
                "Työllisyys, htv": t2["emp_total"] * (1 - d),
            })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fmt = lambda x: f"{x:,.1f}"    # noqa: E731

    print("=" * 96)
    print("YHTEENVETO: KAIKKI LÄHESTYMISTAVAT SAMALLA AKSELILLA")
    print("=" * 96)

    print("\n--- Y.1 Päätulokset ---")
    h = headline()
    print(h.to_string(index=False, float_format=fmt))
    h.to_csv(os.path.join(OUT, "synthesis_headline.csv"), index=False)

    print("\n--- Y.2 Työllisyys ---")
    e = employment_summary()
    print(e.to_string(index=False, float_format=fmt))
    e.to_csv(os.path.join(OUT, "synthesis_employment.csv"), index=False)

    print("\n--- Y.3 Julkiset väitteet vs. tämä analyysi ---")
    c = claims_table()
    print(c.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    c.to_csv(os.path.join(OUT, "synthesis_claims.csv"), index=False)

    print("\n--- Y.4 Vaihteluväli ja sitä ajavat oletukset ---")
    r = range_table()
    print(r.to_string(index=False, float_format=fmt))
    r.to_csv(os.path.join(OUT, "synthesis_range.csv"), index=False)

    lo = r["Arvonlisäys, M€/vuosi"].min()
    hi = r["Arvonlisäys, M€/vuosi"].max()
    print(f"\n  Rakennusvaiheen BKT-vaikutus: {lo:,.0f} - {hi:,.0f} M€/vuosi "
          f"({lo/GDP_2025*100:.2f} - {hi/GDP_2025*100:.2f} % BKT:sta)")


if __name__ == "__main__":
    main()
