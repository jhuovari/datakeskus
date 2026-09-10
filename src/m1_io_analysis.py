"""Approach 1 -- Input-output (Leontief) impact analysis of the construction phase.

Decomposes the €13 bn programme into
  direct   : first-round demand for Finnish output
  indirect : supply-chain rounds, (L - I) f
  induced  : consumption out of the wage income created (Type II closure)

and reports output, value added, compensation, employment and import leakage.
The results are then confronted with Google's own published impact claims.

Units discipline
----------------
The labour coefficient from pt/14yn is *persons per million euro of annual
output*.  Feeding it a demand vector that represents the whole two-year
programme therefore returns **person-years for the whole programme**, not an
annual headcount.  Everything below is reported both ways, because the
distinction is exactly where published "job" figures usually go wrong:
`_total` = over 2027-2028, `_per_year` = divided by the two years.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from io_model import IOSystem            # noqa: E402
import scenario as sc                    # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
GDP_2025 = 281_674.0        # M EUR, StatFin ntp/15a9
N_YEARS = 2.0


def _coef(num: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(num / np.where(x == 0, np.nan, x))


def type2_output(s: IOSystem, f: np.ndarray) -> np.ndarray:
    Ac, _ = s.closed_system()
    Lc = np.linalg.inv(np.eye(s.n + 1) - Ac)
    return (Lc @ np.append(f, 0.0))[: s.n]


def decompose(s: IOSystem, shock: sc.Shock) -> pd.DataFrame:
    """Direct / indirect / induced decomposition, totals over the programme."""
    f = shock.f
    x_direct = f.copy()
    x_typeI = s.L @ f
    x_indirect = x_typeI - x_direct
    x_induced = type2_output(s, f) - x_typeI

    def agg(x):
        return {
            "Tuotos, M€": x.sum(),
            "Arvonlisäys, M€": float(s.v @ x),
            "Palkansaajakorvaukset, M€": float(s.wage_coef @ x),
            "Toimintaylijäämä (netto), M€": float(_coef(s.gos, s.x) @ x),
            "Kiinteän pääoman kuluminen, M€": float(_coef(s.cfc, s.x) @ x),
            "Työllisyys, henkilötyövuotta": float(s.emp_coef @ x),
            "Tuontivälituotteet, M€": float(s.imp_coef @ x),
        }

    df = pd.DataFrame({"Välitön": agg(x_direct),
                       "Välillinen": agg(x_indirect),
                       "Tulokerrannais": agg(x_induced)})
    df["Yhteensä"] = df.sum(axis=1)
    return df


def summary(s: IOSystem, shock: sc.Shock) -> pd.Series:
    t1 = s.impact(shock.f)
    t2 = s.impact(shock.f, type2=True)
    capex = shock.total
    return pd.Series({
        "Ilmoitettu investointi, M€": capex,
        "  josta kotimainen loppukysyntä (perushint.), M€": shock.f.sum(),
        "  josta suora tuonti, M€": shock.direct_imports,
        "  josta tuoteverot, M€": shock.product_taxes,
        "  josta maa ja luvat (ei tuotantoa), M€": shock.transfers,
        "Kotimaisuusaste, %": 100 * shock.domestic_share,
        "Tuotos, Type I, M€": t1["output_total"],
        "Tuotos, Type II, M€": t2["output_total"],
        "Arvonlisäys, Type I, M€": t1["va_total"],
        "Arvonlisäys, Type II, M€": t2["va_total"],
        "Arvonlisäys, Type II, M€/vuosi": t2["va_total"] / N_YEARS,
        "BKT-vaikutus, % BKT:sta vuodessa": t2["va_total"] / N_YEARS / GDP_2025 * 100,
        "Työllisyys, Type I, htv (koko ohjelma)": t1["emp_total"],
        "Työllisyys, Type II, htv (koko ohjelma)": t2["emp_total"],
        "Työllisyys, Type II, henkeä keskimäärin/vuosi": t2["emp_total"] / N_YEARS,
        "Tuonti yhteensä (suora + välituote), M€": shock.direct_imports + t2["imports_induced"],
        "Kerroin: arvonlisäys / ilmoitettu investointi (Type II)": t2["va_total"] / capex,
        "Kerroin: arvonlisäys / kotimainen kysyntä (Type I)": t1["va_total"] / shock.f.sum(),
        "Kerroin: tuotos / kotimainen kysyntä (Type I)": t1["output_total"] / shock.f.sum(),
        "Työllisyys per mrd € investointia, htv (Type II)": t2["emp_total"] / capex * 1000,
        "Työllisyys per mrd € kotimaista kysyntää, htv (Type II)":
            t2["emp_total"] / shock.f.sum() * 1000,
    })


def sector_table(s: IOSystem, shock: sc.Shock, top: int = 15) -> pd.DataFrame:
    x2 = type2_output(s, shock.f)
    df = pd.DataFrame({
        "Toimiala": [s.labels[c] for c in s.industries],
        "Suora kysyntä, M€": shock.f,
        "Tuotos Type I, M€": s.L @ shock.f,
        "Tuotos Type II, M€": x2,
        "Arvonlisäys Type II, M€": s.v * x2,
        "Työllisyys Type II, htv": s.emp_coef * x2,
    }, index=s.industries)
    df["Osuus työllisyysvaik., %"] = 100 * df["Työllisyys Type II, htv"] / df["Työllisyys Type II, htv"].sum()
    return df.sort_values("Tuotos Type II, M€", ascending=False).head(top)


def sensitivity(s: IOSystem) -> pd.DataFrame:
    rows = {}
    for key in ("low_import", "central", "high_import"):
        shock = sc.build_shock(s.industries, it_share_key=key)
        t1, t2 = s.impact(shock.f), s.impact(shock.f, type2=True)
        rows[f"IT-osuus {sc.IT_SHARE[key]:.0%}"] = {
            "Kotimaisuusaste, %": 100 * shock.domestic_share,
            "Kotimainen kysyntä, M€": shock.f.sum(),
            "Arvonlisäys Type I, M€": t1["va_total"],
            "Arvonlisäys Type II, M€": t2["va_total"],
            "Arvonlisäys, M€/vuosi": t2["va_total"] / N_YEARS,
            "BKT-vaikutus, %/vuosi": t2["va_total"] / N_YEARS / GDP_2025 * 100,
            "Työllisyys, htv (koko ohjelma)": t2["emp_total"],
            "Työllisyys, henkeä/vuosi": t2["emp_total"] / N_YEARS,
        }
    return pd.DataFrame(rows)


def compare_with_claims(s: IOSystem) -> pd.DataFrame:
    """Test Google's published construction-phase numbers against the IO framework."""
    shock = sc.build_shock(s.industries)
    t2 = s.impact(shock.f, type2=True)
    x2 = type2_output(s, shock.f)
    iF = s.idx("F")

    # What would 16,000 direct construction jobs *for a year* require?
    direct_F_coef = s.emp[iF] * 1000 / s.x[iF]          # persons per M EUR output
    implied_F_output_per_year = sc.GOOGLE_CLAIMS["jobs_construction"] / direct_F_coef
    our_F_htv = s.emp_coef[iF] * x2[iF]

    # And what capex would be needed to generate 37,000 person-years, at the
    # value-added and labour intensity our scenario implies?
    htv_per_capex = t2["emp_total"] / shock.total
    implied_capex = sc.GOOGLE_CLAIMS["jobs_total"] / htv_per_capex

    rows = [
        ("BKT-vaikutus, M€/vuosi (2027-2028)",
         sc.GOOGLE_CLAIMS["gdp_per_year_meur"], t2["va_total"] / N_YEARS,
         f"suhde {t2['va_total'] / N_YEARS / sc.GOOGLE_CLAIMS['gdp_per_year_meur']:.2f}"),
        ("BKT-vaikutus, % BKT:sta vuodessa",
         sc.GOOGLE_CLAIMS["gdp_per_year_meur"] / GDP_2025 * 100,
         t2["va_total"] / N_YEARS / GDP_2025 * 100, "BKT 2025 = 281,7 mrd €"),
        ("Työllisyys koko maassa",
         sc.GOOGLE_CLAIMS["jobs_total"], t2["emp_total"],
         "Google: 'jobs'; tässä: henkilötyövuotta koko ohjelmalta"),
        ("Työllisyys koko maassa, henkeä/vuosi",
         sc.GOOGLE_CLAIMS["jobs_total"], t2["emp_total"] / N_YEARS,
         "sama jaettuna kahdelle vuodelle"),
        ("Työllisyys rakentamisessa (toimiala F)",
         sc.GOOGLE_CLAIMS["jobs_construction"], our_F_htv,
         "htv koko ohjelmalta, Type II"),
        ("Rakentamisen tuotos, jota 16 000 hengen vuosityöllisyys edellyttäisi, M€/v",
         implied_F_output_per_year, x2[iF] / N_YEARS,
         "vertailu: mallin tuotos F, M€/vuosi"),
        ("Investointi, jota 37 000 htv edellyttäisi tällä kotimaisuusasteella, M€",
         implied_capex, shock.total,
         f"kotimaisuusaste {shock.domestic_share:.0%}"),
    ]
    return pd.DataFrame(rows, columns=["Suure", "Google / implikaatio",
                                       "Tämä analyysi", "Huomautus"])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    fmt = lambda x: f"{x:,.1f}"  # noqa: E731

    print("=" * 96)
    print("LÄHESTYMISTAPA 1: PANOS-TUOTOSANALYYSI -- rakentamisvaihe 2027-2028")
    print("=" * 96)
    print(f"Skenaario: {shock.name}")
    print(f"Panos-tuotostaulukko: Tilastokeskus pt/14yn, vuosi {s.year}, "
          f"{s.n} toimialaa\n")

    m = summary(s, shock)
    print("--- Yhteenveto (koko ohjelma, 13 mrd €, kaksi vuotta) ---")
    for k, v in m.items():
        print(f"  {k:58s} {v:12,.2f}")
    m.to_csv(os.path.join(OUT, "m1_summary.csv"))

    print("\n--- Vaikutusten hajotelma (koko ohjelma) ---")
    d = decompose(s, shock)
    print(d.to_string(float_format=fmt))
    d.to_csv(os.path.join(OUT, "m1_decomposition.csv"))

    print("\n--- Vaikutus toimialoittain (Type II, koko ohjelma) ---")
    st = sector_table(s, shock)
    with pd.option_context("display.max_colwidth", 46, "display.width", 200):
        print(st.to_string(float_format=fmt))
    st.to_csv(os.path.join(OUT, "m1_sectors.csv"))

    print("\n--- Herkkyys: IT-laitteiden osuus investoinnista ---")
    sv = sensitivity(s)
    print(sv.to_string(float_format=fmt))
    sv.to_csv(os.path.join(OUT, "m1_sensitivity.csv"))

    print("\n--- Googlen julkaisemien lukujen testaus ---")
    cc = compare_with_claims(s)
    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print(cc.to_string(index=False, float_format=fmt))
    cc.to_csv(os.path.join(OUT, "m1_claims.csv"), index=False)


if __name__ == "__main__":
    main()
