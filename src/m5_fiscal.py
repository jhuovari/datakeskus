"""Approach 5 -- Public finances: who collects what, and from which base.

Effective tax rates are read off Finland's own tax accounts rather than taken
from statute, because what matters for a demand shock is the marginal take on
wages, consumption and profits, not the headline rates.

The construction phase and the operating phase have completely different fiscal
profiles.  Construction taxes a large temporary wage bill.  Operations taxes
almost no profit -- Google's Finnish entity is a cost centre, and the empirical
record is unambiguous: Tuike Finland Oy paid €8.9 m of corporate tax on €44.4 m
of taxable income in 2022, and nothing at all in 2024-2025 while investing.
What operations do yield is electricity excise and property tax, and the
electricity tax changed dramatically on 1 July 2026 when data centres were
moved from tax class II to class I -- from 0.063 to 2.325 snt/kWh.
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
import m3_growth as m3            # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
GDP_2025 = m1.GDP_2025

# Fiscal cost avoided per person-year moved out of unemployment: basic
# unemployment allowance plus housing and social assistance, net of the tax the
# benefit itself carries.  Deliberately conservative.
UNEMPLOYMENT_COST_PER_PERSON_YEAR = 0.019      # M EUR = EUR 19,000

# Claims in the public debate that this block tests.
SECTOR_CLAIMS = {
    "verotulot_2030_meur_per_year": 400.0,   # FDCA/EK: EUR 400 m/yr from 2030
    "rakennusvaiheen_verot_meur": 1_700.0,   # FDCA/EK: EUR 1.7 bn, construction
    "investointikanta_mrd": 12.0,            # on EUR 12 bn of investment
}


def effective_rates(year: int = 2024) -> pd.Series:
    """Marginal effective rates, measured from the national accounts."""
    t = pd.read_csv(os.path.join(PROC, "na_taxes.csv"))
    t = t[(t.timeperiod_y == year) & (t.contentscode == "ntp-cp")]
    gov = t[t.sektoriluokitus_7_20230101 == "S13"].set_index("verolaji_11_20190101")["value"]
    loc = t[t.sektoriluokitus_7_20230101 == "S1313"].set_index("verolaji_11_20190101")["value"]
    cen = t[t.sektoriluokitus_7_20230101 == "S1311"].set_index("verolaji_11_20190101")["value"]

    hh = pd.read_csv(os.path.join(PROC, "na_household.csv"))
    hh = hh[(hh.contentscode == "ntp-cp") & (hh.timeperiod_y == year)]
    s14 = hh[hh.sektoriluokitus_7_20230101 == "S14"].set_index("taloustoimi_1_20180101")["value"]
    s1 = hh[hh.sektoriluokitus_7_20230101 == "S1"].set_index("taloustoimi_1_20180101")["value"]

    d1 = s1["D1K"]                       # total labour cost in the economy
    d12 = s1["D12K"]                     # employers' social contributions
    wages = d1 - d12
    hh_income_tax = s14["D5K"]
    hh_contrib = s14["D613K"]
    consumption = s14["P3K"]
    vat = gov.get("5111", np.nan)

    return pd.Series({
        "Työnantajan sosiaaliturvamaksut / työvoimakustannus": d12 / d1,
        "Kotitalouksien tulovero / palkat": hh_income_tax / wages,
        "Kotitalouksien sosiaaliturvamaksut / palkat": hh_contrib / wages,
        "Työn kokonaisveroaste (kiila)": (d12 + hh_income_tax + hh_contrib) / d1,
        "ALV / kotitalouksien kulutusmenot": vat / consumption,
        "Yhteisöverokanta (lakisääteinen)": 0.20,
        "Kokonaisveroaste, % BKT:sta": gov["SSS"] / 276_151 * 100,
        "Paikallishallinnon osuus veroista, %": loc["SSS"] / gov["SSS"] * 100,
        "Valtionhallinnon osuus veroista, %": cen["SSS"] / gov["SSS"] * 100,
        "Kiinteistövero koko maassa, M€": gov.get("410001", np.nan),
        "Yhteisövero koko maassa, M€": gov.get("1200", np.nan),
        "Energiaverot koko maassa, M€": gov.get("512102", np.nan),
    })


def construction_phase(unemployment_share: float = 0.5) -> pd.DataFrame:
    """Tax yield of the 2027-2028 construction programme (whole programme)."""
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    dec = m1.decompose(s, shock)
    r = effective_rates()

    labour_cost = dec.loc["Palkansaajakorvaukset, M€", "Yhteensä"]
    employer_contrib = labour_cost * r["Työnantajan sosiaaliturvamaksut / työvoimakustannus"]
    wages = labour_cost - employer_contrib
    income_tax = wages * r["Kotitalouksien tulovero / palkat"]
    employee_contrib = wages * r["Kotitalouksien sosiaaliturvamaksut / palkat"]

    # Induced household consumption drives VAT.  The Type II closure already
    # tells us the household income created; consumption follows from the APC.
    t2 = s.impact(shock.f, type2=True)
    induced_income = t2["household_income"]
    vat = induced_income * s.household_params["apc"] * r["ALV / kotitalouksien kulutusmenot"]

    gos = dec.loc["Toimintaylijäämä (netto), M€", "Yhteensä"]
    corp_tax = gos * 0.20 * 0.6      # 40% of net surplus is unincorporated mixed income

    other_prod_tax = float((s.othertax / np.where(s.x == 0, np.nan, s.x)) @
                           np.nan_to_num(m1.type2_output(s, shock.f)))

    emp = dec.loc["Työllisyys, henkilötyövuotta", "Yhteensä"]
    benefit_saving = emp * unemployment_share * UNEMPLOYMENT_COST_PER_PERSON_YEAR

    rows = [
        ("Tuoteverot investoinnista (ALV, muut)", shock.product_taxes, "valtio"),
        ("Työnantajan sosiaaliturvamaksut", employer_contrib, "sosiaaliturvarahastot"),
        ("Palkansaajien tulovero", income_tax, "valtio + kunnat"),
        ("Palkansaajien sosiaaliturvamaksut", employee_contrib, "sosiaaliturvarahastot"),
        ("ALV kerrannaiskulutuksesta", vat, "valtio"),
        ("Yhteisövero hankintaketjussa", corp_tax, "valtio + kunnat"),
        ("Muut tuotantoverot", other_prod_tax, "valtio + kunnat"),
        (f"Työttömyysmenojen säästö ({unemployment_share:.0%} tulee työttömyydestä)",
         benefit_saving, "valtio + sosiaaliturvarahastot"),
    ]
    df = pd.DataFrame(rows, columns=["Erä", "M€ (koko ohjelma)", "Saaja"])
    df["M€ / vuosi"] = df["M€ (koko ohjelma)"] / m1.N_YEARS
    total = df["M€ (koko ohjelma)"].sum()
    df.loc[len(df)] = ["YHTEENSÄ", total, "", total / m1.N_YEARS]
    return df


def operations_phase() -> pd.DataFrame:
    """Annual tax yield in the steady state, once all four sites run."""
    s = IOSystem.load(2023)
    ops = m3.operations()
    r = effective_rates()

    # Domestic supply chain of operations (same vector as the cascade block).
    f = pd.Series(0.0, index=s.industries)
    f["D35"] = ops["Sähkön osto (energia + siirto), M€/vuosi"]
    f["F"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.5
    f["N80TN82"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.3
    f["J62_J63"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.2
    chain = s.impact(f.to_numpy(), type2=True)

    dc_labour = ops["Palkkakustannus, M€/vuosi"]
    all_labour = dc_labour + chain["comp_total"]
    employer_contrib = all_labour * r["Työnantajan sosiaaliturvamaksut / työvoimakustannus"]
    wages = all_labour - employer_contrib
    income_tax = wages * r["Kotitalouksien tulovero / palkat"]
    employee_contrib = wages * r["Kotitalouksien sosiaaliturvamaksut / palkat"]
    vat = chain["household_income"] * s.household_params["apc"] * r["ALV / kotitalouksien kulutusmenot"]

    rows = [
        ("Sähkövero (veroluokka I, 2,325 snt/kWh)",
         ops["Sähkövero, M€/vuosi"], "valtio"),
        ("Kiinteistövero", ops["Kiinteistövero, M€/vuosi"], "kunnat"),
        ("Yhteisövero, datakeskusyhtiö", ops["Yhteisövero, M€/vuosi"], "valtio + kunnat"),
        ("Yhteisövero, hankintaketju", chain["gos_total"] * 0.20 * 0.6, "valtio + kunnat"),
        ("Työnantajan sosiaaliturvamaksut", employer_contrib, "sosiaaliturvarahastot"),
        ("Palkansaajien tulovero", income_tax, "valtio + kunnat"),
        ("Palkansaajien sosiaaliturvamaksut", employee_contrib, "sosiaaliturvarahastot"),
        ("ALV kerrannaiskulutuksesta", vat, "valtio"),
        ("Muut tuotantoverot hankintaketjussa", chain["othertax_total"], "valtio + kunnat"),
    ]
    df = pd.DataFrame(rows, columns=["Erä", "M€ / vuosi", "Saaja"])
    total = df["M€ / vuosi"].sum()
    df.loc[len(df)] = ["YHTEENSÄ", total, ""]
    df["% BKT:sta"] = df["M€ / vuosi"] / GDP_2025 * 100
    return df


def electricity_tax_counterfactual() -> pd.DataFrame:
    """What the 1.7.2026 tax-class change is worth on this load."""
    ops = m3.operations()
    twh = ops["Sähkönkulutus, TWh/vuosi"]
    rows = []
    for label, rate in (("Veroluokka II (voimassa 30.6.2026 asti)", 0.135),
                        ("Veroluokka I (voimassa 1.7.2026 alkaen)", 2.325)):
        rows.append({"Veroluokka": label, "snt/kWh": rate,
                     "Verotuotto, M€/vuosi": twh * 1e9 * rate / 100 / 1e6})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["Muutoksen arvo valtiolle", np.nan,
                       df["Verotuotto, M€/vuosi"].iloc[1] - df["Verotuotto, M€/vuosi"].iloc[0]]
    return df


def compare_sector_claims(constr: pd.DataFrame, ops_tax: pd.DataFrame) -> pd.DataFrame:
    """Scale the sector's published claims to this project and compare."""
    scale = sc.TOTAL_CAPEX / 1000 / SECTOR_CLAIMS["investointikanta_mrd"]
    ours_c = constr.loc[constr["Erä"] == "YHTEENSÄ", "M€ (koko ohjelma)"].iloc[0]
    ours_o = ops_tax.loc[ops_tax["Erä"] == "YHTEENSÄ", "M€ / vuosi"].iloc[0]
    rows = [
        ("Rakennusvaiheen verot, M€ (koko ohjelma)",
         SECTOR_CLAIMS["rakennusvaiheen_verot_meur"] * scale, ours_c,
         f"toimialan väite skaalattu {scale:.2f}x (13 mrd / 12 mrd)"),
        ("Käyttövaiheen verot, M€/vuosi",
         SECTOR_CLAIMS["verotulot_2030_meur_per_year"] * scale, ours_o,
         "toimialan väite koskee kaikkia Suomen datakeskuksia"),
    ]
    return pd.DataFrame(rows, columns=["Suure", "Toimialan väite (skaalattu)",
                                       "Tämä analyysi", "Huomautus"])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fmt = lambda x: f"{x:,.2f}"    # noqa: E731

    print("=" * 96)
    print("LÄHESTYMISTAPA 5: JULKINEN TALOUS")
    print("=" * 96)

    print("\n--- 5.1 Efektiiviset veroasteet kansantalouden tilinpidosta (2024) ---")
    r = effective_rates()
    for k, v in r.items():
        print(f"  {k:52s} {v:12,.4f}")
    r.to_csv(os.path.join(OUT, "m5_rates.csv"))

    print("\n--- 5.2 Rakennusvaihe 2027-2028 ---")
    c = construction_phase()
    print(c.to_string(index=False, float_format=fmt))
    c.to_csv(os.path.join(OUT, "m5_construction.csv"), index=False)

    print("\n--- 5.3 Käyttövaihe, vuositasolla ---")
    o = operations_phase()
    print(o.to_string(index=False, float_format=fmt))
    o.to_csv(os.path.join(OUT, "m5_operations.csv"), index=False)

    print("\n--- 5.4 Sähköveroluokan muutos 1.7.2026 ---")
    et = electricity_tax_counterfactual()
    print(et.to_string(index=False, float_format=fmt))
    et.to_csv(os.path.join(OUT, "m5_electricity_tax.csv"), index=False)

    print("\n--- 5.5 Toimialan julkisten väitteiden vertailu ---")
    cmp = compare_sector_claims(c, o)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(cmp.to_string(index=False, float_format=fmt))
    cmp.to_csv(os.path.join(OUT, "m5_claims.csv"), index=False)

    print("\n--- 5.6 Herkkyys: kuinka suuri osa työvoimasta tulee työttömyydestä ---")
    rows = []
    for share in (0.0, 0.25, 0.5, 0.75):
        cc = construction_phase(unemployment_share=share)
        rows.append({"Työttömyydestä tuleva osuus": share,
                     "Verotuotto + säästöt, M€": cc.loc[cc["Erä"] == "YHTEENSÄ",
                                                        "M€ (koko ohjelma)"].iloc[0]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
