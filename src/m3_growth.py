"""Approach 3 -- Supply side: capital deepening, and the GDP/GNI wedge.

Two questions the demand-side blocks cannot answer.

**How much does Finland's productive capacity actually rise?**  A neoclassical
growth-accounting calculation on the capital stock, using depreciation rates
read off Finland's own national accounts by asset type.

**Who gets the income?**  This is where a foreign-owned, capital-intensive,
fast-depreciating asset behaves very differently from a domestic factory.  Once
the campus is running, the largest single component of the value added it
records in Finland is *consumption of fixed capital* on equipment owned by a
non-resident group.  That figure is in GDP.  It is not income to anybody in
Finland.  So we run the full cascade

    GDP  ->  GNI (less primary income paid abroad)  ->  NNI (less depreciation)

which is exactly the correction Ireland had to invent GNI* for.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from io_model import IOSystem      # noqa: E402
import scenario as sc              # noqa: E402
import m1_io_analysis as m1        # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
GDP_2025 = m1.GDP_2025

# ---------------------------------------------------------------- parameters
# Economic depreciation.  Building and machinery rates come from Finland's own
# capital accounts (P51CK / NKANTA by asset type, StatFin ntp/15af, 2024).
# AI servers are the exception: hyperscalers depreciate server fleets over
# 5-6 years, and GPU generations turn over faster still, so we set the IT rate
# from useful life rather than from the economy-wide machinery average.
IT_USEFUL_LIFE_YEARS = {"fast": 4.0, "central": 5.5, "slow": 7.0}

# Physical capacity implied by the capex, per MW of IT load.
IT_CAPEX_PER_MW = {"low": 25.0, "central": 30.0, "high": 40.0}   # M EUR / MW
PUE = 1.15              # power usage effectiveness, modern liquid-cooled AI hall
UTILISATION = 0.90      # AI training/inference clusters run near flat out

# Operations-phase staffing, anchored on Google's own Hamina site: about 500
# people on site after roughly €4.5 bn of cumulative investment (Yle, 2025).
STAFF_PER_BN_EUR = {"low": 60.0, "central": 95.0, "high": 130.0}

# Cost-plus markup on the Finnish service entity.  Anchor: Tuike Finland Oy
# (the Google data-centre company in Hamina) reported taxable income of
# €44.4 m in 2022 against an asset base of roughly €3.5 bn.
MARKUP = {"low": 0.03, "central": 0.05, "high": 0.10}

CORPORATE_TAX_RATE = 0.20
ELECTRICITY_PRICE_EUR_MWH = 55.0      # PPA-weighted wholesale, 2027-2028
GRID_FEE_EUR_MWH = 8.0                # transmission tariff, large industrial
ELECTRICITY_TAX_EUR_MWH = 23.25       # tax class I from 1.4.2026: 2.325 snt/kWh
MAINTENANCE_SHARE_OF_FACILITY = 0.03  # annual O&M as share of facility capex


def depreciation_rates(year: int = 2024) -> pd.Series:
    """Economic depreciation by asset type, measured from the capital accounts."""
    c = pd.read_csv(os.path.join(PROC, "na_capital.csv"))
    c = c[(c.timeperiod_y == year) & (c.contentscode == "ntp-cp")]
    piv = c.pivot_table(index="varojenluokitus_5_20180101",
                        columns="taloustoimi_1_20180101", values="value")
    d = (piv["P51CK"] / piv["NKANTA"]).dropna()
    return d.rename("delta_net")


def capital_composition(it_share_key: str = "central",
                        it_life: float | None = None) -> pd.DataFrame:
    """The €13 bn split into asset classes with their depreciation rates."""
    deltas = depreciation_rates()
    it_life = it_life if it_life is not None else IT_USEFUL_LIFE_YEARS["central"]
    it_share = sc.IT_SHARE[it_share_key]
    facility = sc.TOTAL_CAPEX * (1 - it_share)

    rows = [
        # asset, amount, delta, basis
        ("IT-laitteet (palvelimet, kiihdyttimet, verkko)",
         sc.TOTAL_CAPEX * it_share, 1 / it_life,
         f"pitoaika {it_life:.1f} v"),
        ("Sähköjärjestelmät (muuntajat, UPS, varavoima)",
         facility * sc.FACILITY_SPLIT["electrical"], float(deltas["N1132_N1139"]) * 0.55,
         "koneet ja laitteet, pidempi pitoaika"),
        ("Jäähdytys ja LVI",
         facility * sc.FACILITY_SPLIT["mechanical"], float(deltas["N1132_N1139"]) * 0.55,
         "koneet ja laitteet, pidempi pitoaika"),
        ("Rakennukset ja rakennelmat",
         facility * sc.FACILITY_SPLIT["shell_civil"], float(deltas["N1121"]),
         "TK: muut talorakennukset"),
        ("Suunnittelu, aktivoitu",
         facility * sc.FACILITY_SPLIT["engineering"], float(deltas["N1121"]),
         "aktivoidaan rakennukseen"),
        ("Maa-alueet ja luvat",
         facility * sc.FACILITY_SPLIT["land_other"], 0.0, "ei poistoja"),
    ]
    df = pd.DataFrame(rows, columns=["Varalaji", "Määrä, M€", "Poistoaste δ", "Peruste"])
    df["Poistot, M€/vuosi"] = df["Määrä, M€"] * df["Poistoaste δ"]
    return df


def growth_accounting() -> pd.Series:
    """Capital-deepening contribution to potential output.

        ΔY/Y = α · ΔK/K

    with α the capital elasticity, proxied by the capital share of value added.
    Reported both against the whole capital stock and, for context, against the
    non-residential stock the investment actually belongs to.
    """
    s = IOSystem.load(2023)
    c = pd.read_csv(os.path.join(PROC, "na_capital.csv"))
    c = c[(c.timeperiod_y == 2024) & (c.taloustoimi_1_20180101 == "NKANTA")
          & (c.contentscode == "ntp-cp")]
    stock = c.set_index("varojenluokitus_5_20180101")["value"]

    labour_share = s.comp.sum() / s.va.sum()
    # Mixed income of the self-employed is part labour, part capital.  Attribute
    # it in the same proportion as the rest of the economy (standard practice).
    alpha_raw = 1 - labour_share
    gva = s.va.sum()
    mixed = s.gos.sum()
    alpha_adj = 1 - (s.comp.sum() + mixed * labour_share) / gva

    k_total = float(stock["N0"])
    k_nonres = float(stock["N0"] - stock["N111"])
    dk = sc.TOTAL_CAPEX

    return pd.Series({
        "Nettopääomakanta 2024, M€": k_total,
        "  josta muu kuin asuinrakennukset, M€": k_nonres,
        "Investointi, M€": dk,
        "ΔK/K, koko kanta, %": dk / k_total * 100,
        "ΔK/K, ei-asuinkanta, %": dk / k_nonres * 100,
        "Palkkojen osuus arvonlisäyksestä": labour_share,
        "Pääoman jousto α (yksinkertainen)": alpha_raw,
        "Pääoman jousto α (sekatulo jaettu)": alpha_adj,
        "Potentiaalisen tuotannon nousu, % (α yksink.)": alpha_raw * dk / k_total * 100,
        "Potentiaalisen tuotannon nousu, % (α korj.)": alpha_adj * dk / k_total * 100,
        "  vastaa BKT:ssa, M€": alpha_adj * dk / k_total * GDP_2025,
    })


def operations(it_share_key: str = "central",
               it_life: float | None = None,
               markup: float | None = None,
               capex_per_mw: float | None = None,
               staff_per_bn: float | None = None) -> pd.Series:
    """Steady-state annual accounts of the operating campus.

    Every uncertain parameter is a separate argument so that sensitivity runs
    vary one thing at a time rather than moving three at once.
    """
    it_life = it_life if it_life is not None else IT_USEFUL_LIFE_YEARS["central"]
    markup = markup if markup is not None else MARKUP["central"]
    capex_per_mw = capex_per_mw if capex_per_mw is not None else IT_CAPEX_PER_MW["central"]
    staff_per_bn = staff_per_bn if staff_per_bn is not None else STAFF_PER_BN_EUR["central"]

    comp = capital_composition(it_share_key, it_life=it_life)
    cfc = comp["Poistot, M€/vuosi"].sum()
    it_capex = sc.TOTAL_CAPEX * sc.IT_SHARE[it_share_key]
    facility_capex = sc.TOTAL_CAPEX * (1 - sc.IT_SHARE[it_share_key])

    # Physical scale
    it_mw = it_capex / capex_per_mw
    twh = it_mw * PUE * 8760 * UTILISATION / 1e6

    # Labour: staff and their cost.  Google claims wages 24% above the median.
    staff = sc.TOTAL_CAPEX / 1000 * staff_per_bn
    median_month = median_wage()
    wage_per_head = median_month * 12 * (1 + sc.GOOGLE_CLAIMS["wage_premium_vs_median"])
    employer_contrib = 0.20
    labour_cost = staff * wage_per_head * (1 + employer_contrib) / 1e6   # M EUR

    # Intermediate purchases
    electricity = twh * 1000 * (ELECTRICITY_PRICE_EUR_MWH + GRID_FEE_EUR_MWH) / 1e3
    electricity_tax = twh * 1000 * ELECTRICITY_TAX_EUR_MWH / 1e3
    maintenance = facility_capex * MAINTENANCE_SHARE_OF_FACILITY
    property_tax = property_tax_estimate(comp)

    intermediates = electricity + maintenance
    costs = cfc + labour_cost + intermediates + electricity_tax + property_tax
    nos = costs * markup
    output = costs + nos
    value_added = output - intermediates
    corp_tax = max(nos, 0.0) * CORPORATE_TAX_RATE
    primary_income_abroad = nos - corp_tax

    # Replacement investment: the IT fleet has to be bought again every few
    # years, and almost all of it is imported.
    replacement = it_capex / it_life

    return pd.Series({
        "IT-teho, MW": it_mw,
        "Sähkönkulutus, TWh/vuosi": twh,
        "Henkilöstö, henkeä": staff,
        "Mediaanipalkka, €/kk": median_month,
        "Palkkakustannus, M€/vuosi": labour_cost,
        "Sähkön osto (energia + siirto), M€/vuosi": electricity,
        "Sähkövero, M€/vuosi": electricity_tax,
        "Kunnossapito ja muut ostot, M€/vuosi": maintenance,
        "Kiinteistövero, M€/vuosi": property_tax,
        "Kiinteän pääoman kuluminen (poistot), M€/vuosi": cfc,
        "Kustannukset yhteensä, M€/vuosi": costs,
        "Kustannuslisä (markup)": markup,
        "Toimintaylijäämä, netto, M€/vuosi": nos,
        "Tuotos (palvelumaksu konsernille), M€/vuosi": output,
        "Arvonlisäys Suomessa, M€/vuosi": value_added,
        "  josta poistoja, %": cfc / value_added * 100,
        "Yhteisövero, M€/vuosi": corp_tax,
        "Ensitulon vienti ulkomaille, M€/vuosi": primary_income_abroad,
        "IT-laitteiden korvausinvestointi, M€/vuosi": replacement,
    })


def median_wage(year: int = 2024) -> float:
    """Median monthly earnings of full-time employees, all industries."""
    p = pd.read_csv(os.path.join(PROC, "earnings_structure.csv"))
    med = p[(p.contentscode == "kokonaisansio_mediaani") |
            (p.label_contentscode.str.contains("kokonaisansion mediaani", na=False))]
    tot = med[(med.toimiala_79_20180101 == "SSS") &
              (med.sukupuoli_9_20180101 == "SSS")]
    if len(tot):
        v = tot["value"].dropna()
        if len(v):
            return float(v.iloc[0])
    # Fall back on the mean from the earnings index table.
    e = pd.read_csv(os.path.join(PROC, "earnings_level.csv"))
    return float(e[e.timeperiod_y == year]["value"].iloc[0]) * 0.92


def property_tax_estimate(comp: pd.DataFrame) -> float:
    """Property tax on the campus.

    Finnish property tax reaches land, buildings and structures only.  Machinery
    and equipment that is not valued as part of the building -- which is exactly
    what servers, and separately housed transformers, are -- fall outside the
    base (Tax Administration guidance on the Real Estate Tax Act).  The base is
    therefore a small fraction of a €13 bn AI campus.
    """
    buildings = comp.loc[comp.Varalaji.str.startswith("Rakennukset"), "Määrä, M€"].sum()
    engineering = comp.loc[comp.Varalaji.str.startswith("Suunnittelu"), "Määrä, M€"].sum()
    land = comp.loc[comp.Varalaji.str.startswith("Maa"), "Määrä, M€"].sum()
    # Taxable value of new buildings is set below construction cost, and the
    # general rate for non-residential buildings runs 0.93-2.00%.
    taxable_building_value = (buildings + engineering) * 0.75
    rate_buildings = 0.0150
    rate_land = 0.0130
    return taxable_building_value * rate_buildings + land * rate_land


def gdp_gni_cascade(ops: pd.Series) -> pd.DataFrame:
    """From headline GDP to what Finnish residents actually earn."""
    s = IOSystem.load(2023)

    # Domestic supply chain of the operating phase: electricity, maintenance
    # and services bought each year in Finland.
    f = pd.Series(0.0, index=s.industries)
    f["D35"] = ops["Sähkön osto (energia + siirto), M€/vuosi"]
    f["F"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.5
    f["N80TN82"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.3
    f["J62_J63"] = ops["Kunnossapito ja muut ostot, M€/vuosi"] * 0.2
    chain = s.impact(f.to_numpy(), type2=True)

    dc_va = ops["Arvonlisäys Suomessa, M€/vuosi"]
    cfc = ops["Kiinteän pääoman kuluminen (poistot), M€/vuosi"]
    nos_abroad = ops["Ensitulon vienti ulkomaille, M€/vuosi"]

    gdp = dc_va + chain["va_total"]
    gni = gdp - nos_abroad
    nni = gni - cfc - chain["cfc_total"]

    rows = [
        ("Datakeskusyhtiön arvonlisäys", dc_va),
        ("Kotimaisen hankintaketjun arvonlisäys (Type II)", chain["va_total"]),
        ("= BKT-vaikutus", gdp),
        ("./. ensitulon vienti ulkomaille (voitot)", -nos_abroad),
        ("= BKTL-vaikutus (GNI)", gni),
        ("./. kiinteän pääoman kuluminen, datakeskus", -cfc),
        ("./. kiinteän pääoman kuluminen, hankintaketju", -chain["cfc_total"]),
        ("= Nettokansantulo-vaikutus (NNI)", nni),
    ]
    df = pd.DataFrame(rows, columns=["Erä", "M€/vuosi"])
    df["% BKT:sta"] = df["M€/vuosi"] / GDP_2025 * 100
    df["Kumulatiivinen"] = ["" if e.startswith(("./.", "Datakeskus", "Kotimaisen"))
                            else "kyllä" for e in df["Erä"]]

    detail = pd.Series({
        "Hankintaketjun työllisyys, htv/vuosi": chain["emp_total"],
        "Datakeskuksen oma henkilöstö, henkeä": ops["Henkilöstö, henkeä"],
        "Työllisyys yhteensä, henkeä": chain["emp_total"] + ops["Henkilöstö, henkeä"],
        "Poistojen osuus BKT-vaikutuksesta, %": cfc / gdp * 100,
        "NNI-vaikutus / BKT-vaikutus, %": nni / gdp * 100,
    })
    return df, detail


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fmt = lambda x: f"{x:,.2f}"     # noqa: E731

    print("=" * 96)
    print("LÄHESTYMISTAPA 3: TARJONTAPUOLI, PÄÄOMAKANTA JA BKT vs. KANSANTULO")
    print("=" * 96)

    print("\n--- 3.1 Poistoasteet Suomen pääomatilinpidosta (2024) ---")
    d = depreciation_rates()
    for k in ["N0", "N1121", "N1122", "N113", "N1132_N1139", "N117"]:
        print(f"  {k:14s} δ (netto) = {d[k]:.3f}")

    print("\n--- 3.2 Investoinnin varalajijakauma ja poistot ---")
    comp = capital_composition()
    print(comp.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    comp.to_csv(os.path.join(OUT, "m3_capital.csv"), index=False)
    print(f"  Poistot yhteensä: {comp['Poistot, M€/vuosi'].sum():,.0f} M€/vuosi "
          f"({comp['Poistot, M€/vuosi'].sum()/sc.TOTAL_CAPEX*100:.1f} % investoinnista)")

    print("\n--- 3.3 Kasvulaskenta: potentiaalisen tuotannon nousu ---")
    ga = growth_accounting()
    for k, v in ga.items():
        print(f"  {k:52s} {v:12,.3f}")
    ga.to_csv(os.path.join(OUT, "m3_growth_accounting.csv"))

    print("\n--- 3.4 Käyttövaiheen tilinpito (vakaa tila) ---")
    ops = operations()
    for k, v in ops.items():
        print(f"  {k:52s} {v:12,.2f}")
    ops.to_csv(os.path.join(OUT, "m3_operations.csv"))

    print("\n--- 3.5 BKT -> BKTL -> nettokansantulo ---")
    casc, det = gdp_gni_cascade(ops)
    print(casc.to_string(index=False, float_format=fmt))
    casc.to_csv(os.path.join(OUT, "m3_cascade.csv"), index=False)
    print()
    for k, v in det.items():
        print(f"  {k:52s} {v:12,.1f}")
    det.to_csv(os.path.join(OUT, "m3_cascade_detail.csv"))

    print("\n--- 3.6 Herkkyys: yksi parametri kerrallaan ---")
    print("  Huomaa: BKT-vaikutus riippuu voimakkaasti poistoaste-oletuksesta,")
    print("  nettokansantulo ei käytännössä lainkaan. Se on tämän analyysin ydinhavainto.")

    def row(label, o):
        c, _ = gdp_gni_cascade(o)
        pick = lambda pre: c.loc[c["Erä"].str.startswith(pre), "M€/vuosi"].iloc[0]  # noqa: E731
        gdp = pick("= BKT")
        nni = pick("= Nettokansantulo")
        return {
            "Skenaario": label,
            "Poistot, M€/v": o["Kiinteän pääoman kuluminen (poistot), M€/vuosi"],
            "IT-teho, MW": o["IT-teho, MW"],
            "Sähkö, TWh/v": o["Sähkönkulutus, TWh/vuosi"],
            "BKT, M€/v": gdp,
            "BKTL, M€/v": pick("= BKTL"),
            "NNI, M€/v": nni,
            "NNI / BKT, %": nni / gdp * 100,
        }

    rows = [row("Perusskenaario", operations())]
    for k, v in IT_USEFUL_LIFE_YEARS.items():
        rows.append(row(f"IT-pitoaika {v:.1f} v", operations(it_life=v)))
    for k, v in MARKUP.items():
        rows.append(row(f"Kustannuslisä {v:.0%}", operations(markup=v)))
    for k, v in IT_CAPEX_PER_MW.items():
        rows.append(row(f"IT-capex {v:.0f} M€/MW", operations(capex_per_mw=v)))
    for k, v in STAFF_PER_BN_EUR.items():
        rows.append(row(f"Henkilöstö {v:.0f}/mrd €", operations(staff_per_bn=v)))
    for k in sc.IT_SHARE:
        rows.append(row(f"IT-osuus {sc.IT_SHARE[k]:.0%}", operations(it_share_key=k)))

    sens = pd.DataFrame(rows).drop_duplicates(subset="Skenaario")
    print(sens.to_string(index=False, float_format=fmt))
    sens.to_csv(os.path.join(OUT, "m3_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
