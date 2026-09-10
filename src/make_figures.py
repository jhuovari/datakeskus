"""Produce every chart in the report from the model output."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import figures as fg              # noqa: E402
from io_model import IOSystem     # noqa: E402
import scenario as sc             # noqa: E402
import m1_io_analysis as m1       # noqa: E402
import m3_growth as m3            # noqa: E402
import m4_electricity as m4       # noqa: E402
import m5_fiscal as m5            # noqa: E402
import m6_synthesis as m6         # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main() -> None:
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    t1 = s.impact(shock.f)
    t2 = s.impact(shock.f, type2=True)
    made = []

    # 1. Where the €13 bn goes -------------------------------------------------
    # Exact identity, verified against the table: domestic final demand at basic
    # prices = value added (Type I) + imported intermediates + product taxes
    # levied along the chain.
    x1 = s.L @ shock.f
    imp_chain = float(s.imp_coef @ x1)
    ptax_chain = shock.f.sum() - t1["va_total"] - imp_chain
    made.append(fg.waterfall(
        "fig1_vuoto.svg",
        "Mihin 13 miljardia euroa menee?",
        "Ilmoitetusta investoinnista kotimaiseen arvonlisäykseen, koko ohjelma 2027-2028, milj. €",
        [("Ilmoitettu investointi", sc.TOTAL_CAPEX, "start"),
         ("Tuontilaitteet ja -palvelut", -shock.direct_imports, "delta"),
         ("Tuoteverot", -shock.product_taxes, "delta"),
         ("Maa ja luvat", -shock.transfers, "delta"),
         ("Kotimainen loppukysyntä", shock.f.sum(), "total"),
         ("Välituotetuonti ketjussa", -imp_chain, "delta"),
         ("Tuoteverot ketjussa", -ptax_chain, "delta"),
         ("Arvonlisäys, Type I", t1["va_total"], "total")],
        note=f"Type II lisää tulokerrannaisvaikutuksen "
             f"{t2['va_total'] - t1['va_total']:,.0f} M€, jolloin arvonlisäys on "
             f"{t2['va_total']:,.0f} M€. Lähde: Tilastokeskus pt/14yn (2023)."))

    # 2. GDP -> GNI -> NNI ----------------------------------------------------
    ops = m3.operations()
    casc, det = m3.gdp_gni_cascade(ops)
    pick = lambda pre: float(casc.loc[casc["Erä"].str.startswith(pre), "M€/vuosi"].iloc[0])  # noqa: E731
    made.append(fg.waterfall(
        "fig2_bkt_kansantulo.svg",
        "Käyttövaihe: BKT nousee, kansantulo ei juuri",
        "Vuositasolla vakaassa tilassa, milj. €",
        [("BKT-vaikutus", pick("= BKT"), "start"),
         ("Voitot ulkomaille", -abs(pick("./. ensitulon")), "delta"),
         ("BKTL (GNI)", pick("= BKTL"), "total"),
         ("Poistot, datakeskus", -abs(pick("./. kiinteän pääoman kuluminen, datakeskus")), "delta"),
         ("Poistot, ketju", -abs(pick("./. kiinteän pääoman kuluminen, hankintaketju")), "delta"),
         ("Nettokansantulo", pick("= Nettokansantulo"), "total")],
        note="Poistot ulkomaisen omistajan laitteista ovat BKT:ssa mutta eivät kenenkään tuloa Suomessa."))

    # 3. Claims vs analysis ---------------------------------------------------
    cl = m6.claims_table()
    money = cl[cl["Suure"].str.contains("M€")]
    jobs = cl[~cl["Suure"].str.contains("M€")]
    made.append(fg.grouped_bars(
        "fig3_vaitteet_raha.svg",
        "Julkiset väitteet ja virallisiin tilastoihin perustuva arvio",
        "Rahamääräiset suureet, milj. €",
        list(money["Suure"]),
        [("Julkinen väite", list(money["Julkinen väite"])),
         ("Tämä analyysi", list(money["Tämä analyysi"]))],
        note="Google 9.9.2026; FDCA/EK skaalattu 13/12 mrd €. Analyysi: Tilastokeskuksen panos-tuotosaineisto."))
    made.append(fg.grouped_bars(
        "fig4_vaitteet_tyollisyys.svg",
        "Työllisyysväitteet: 'työpaikkoja' vai henkilötyövuosia?",
        "Henkilöä tai henkilötyövuotta",
        list(jobs["Suure"]),
        [("Julkinen väite", list(jobs["Julkinen väite"])),
         ("Tämä analyysi", list(jobs["Tämä analyysi"]))],
        note="Google ilmoittaa 'jobs'; tämä analyysi henkilötyövuosia. Ero on osin yksikkökysymys."))

    # 4. Sector pattern -------------------------------------------------------
    st = m1.sector_table(s, shock, top=14)
    short = {"F": "Rakentaminen", "M71": "Arkkitehti- ja insinööripalvelut",
             "C25": "Metallituotteet", "G46": "Tukkukauppa",
             "J62_J63": "Tietojenkäsittelypalvelu", "C27": "Sähkölaitteet",
             "M69_M70": "Liikkeenjohdon palvelut", "G47": "Vähittäiskauppa",
             "H49": "Maaliikenne", "C33": "Koneiden asennus ja huolto",
             "C28": "Muut koneet ja laitteet", "H52": "Varastointi ja logistiikka",
             "C16": "Sahatavara ja puutuotteet", "I": "Majoitus ja ravitsemis",
             "L68XL68202": "Kiinteistöala", "N80TN82": "Muut tukipalvelut"}
    rows = sorted(
        [(short.get(c, str(s.labels[c]).split(" ", 1)[-1][:36]), float(v))
         for c, v in st["Työllisyys Type II, htv"].items() if v > 1.0],
        key=lambda r: -r[1])[:10]
    made.append(fg.hbar(
        "fig5_toimialat.svg",
        "Työllisyysvaikutus keskittyy rakentamiseen ja suunnitteluun",
        "Henkilötyövuotta koko ohjelmalta, Type II, 10 suurinta toimialaa",
        rows, unit=" htv", colours=[fg.SERIES[0]] * 10,
        note="Rakentaminen (F) 30 % ja tekniset palvelut (71) 10 % koko työllisyysvaikutuksesta."))

    # 5. Robustness: GDP moves, national income does not ----------------------
    sens = pd.read_csv(os.path.join(os.path.dirname(__file__), "..",
                                    "output", "tables", "m3_sensitivity.csv"))
    sens = sens[~sens.Skenaario.str.startswith("Perus")]
    made.append(fg.dot_compare(
        "fig6_robustisuus.svg",
        "BKT-vaikutus riippuu oletuksista, nettokansantulo ei",
        "Käyttövaihe, milj. €/vuosi. Yksi parametri muutettuna kerrallaan.",
        [(r.Skenaario, float(r["BKT, M€/v"]), float(r["NNI, M€/v"]))
         for _, r in sens.iterrows()],
        labels=("BKT-vaikutus", "Nettokansantulo"), unit=" M€",
        note="BKT-vaikutus vaihtelee 2 000-3 000 M€, nettokansantulo pysyy 345-419 M€:ssa."))

    # 6. Construction slack ---------------------------------------------------
    emp = pd.read_csv(os.path.join(PROC, "na_employment.csv"))
    con = emp[(emp.toimiala_79_20180101 == "F") &
              (emp.taloustoimi_1_20180101 == "E1")].set_index("timeperiod_y")["value"]
    con = con.loc[2000:2025]
    peak = con.max()
    made.append(fg.line_chart(
        "fig7_rakentamisen_vaje.svg",
        "Rakennusala on syvässä taantumassa -- siksi tilaa on",
        "Rakentamisen työlliset, 1000 henkeä",
        [str(y) for y in con.index], [("Rakentaminen", list(con.values))],
        note="Vaje huipusta (2018) 28 000 henkeä. Hankkeen rakennusalan työvoimatarve "
             "on n. 4 100 henkeä vuodessa.",
        hlines=[(float(peak), f"huippu 2018: {peak:.0f}")]))

    # 7. Electricity ----------------------------------------------------------
    sd = m4.supply_demand()
    made.append(fg.hbar(
        "fig8_sahko.svg",
        "Vuositase riittää, tuntitase ei",
        "Sähkön vuositase, TWh vuodessa",
        [("Datakeskusten kulutus, TWh", float(sd["Datakeskusten sähkönkulutus, TWh/vuosi"])),
         ("Uusi sopimustuulivoima, TWh", float(sd["  tuotanto, TWh/vuosi (CF 0.36)"])),
         ("Jäännöskysyntä vuositasolla, TWh",
          float(sd["Vuositason jäännöskysyntä (kulutus - uusi tuuli), TWh"]))],
        colours=[fg.SERIES[0], fg.SERIES[2], fg.SERIES[1]],
        note="Heikkotuulisina tunteina tuuli tuottaa n. 44 MW, kun kuorma on 305 MW: "
             "86 % kuormasta jää muun järjestelmän katettavaksi."))

    for p in made:
        print("kirjoitettu", os.path.relpath(p))


if __name__ == "__main__":
    main()
