"""Approach 4 -- The electricity market: how binding is the energy constraint?

A 300 MW AI campus is, for the Finnish power system, a new industrial region.
Google's answer is "bring your own power": 629 MW of new-to-grid onshore wind,
a 22-year PPA tied to the Loviisa life extension, and a 94 MW battery.

Two things have to be kept apart.

**Annual energy.** Contracted new wind roughly covers the campus's annual
consumption, so on a yearly balance the addition is close to neutral.

**Hourly capacity.** An AI campus is flat baseload; wind is not.  In the
low-wind hours -- which are exactly the price-setting hours -- almost the whole
load falls on the rest of the system.  Annual matching does not remove the
price effect; it only removes it from the annual average.

The price effect is then priced with a residual-demand calculation across a
range of short-run supply elasticities, and the result is decomposed into a
transfer from consumers to producers versus a genuine efficiency loss.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import scenario as sc            # noqa: E402
import m3_growth as m3           # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")

WIND_CF = 0.36              # capacity factor, new Finnish onshore wind
NUCLEAR_CF = 0.92
LOVIISA_MW = 1_020.0        # two units, net
LOVIISA_PPA_SHARE = 0.50    # "up to 50% of the plant's output"

SUPPLY_ELASTICITIES = (0.3, 0.5, 1.0)
# Share of hours in which the wind fleet delivers close to nothing, and the
# output it delivers then.  Finnish onshore wind spends roughly a tenth of the
# year below 10% of nameplate.
LOW_WIND_HOUR_SHARE = 0.10
LOW_WIND_OUTPUT_FRACTION = 0.07


def electricity_balance() -> pd.Series:
    e = pd.read_csv(os.path.join(PROC, "electricity.csv"))
    e = e[(e.contentscode == "maara_gwh") & (e.timeperiod_y == 2024)]
    b = e.set_index("energia_30_20200915")["value"]
    s = pd.read_csv(os.path.join(PROC, "electricity_sector.csv"))
    s = s[(s.contentscode == "maara_gwh") & (s.timeperiod_y == 2024)]
    sec = s.set_index("energia_34_20201002")["value"]
    return pd.Series({
        "Sähkön kokonaiskulutus 2024, GWh": b["SSS"],
        "Kotimainen tuotanto, GWh": b["1"],
        "Nettotuonti, GWh": b["2"],
        "Tuulivoima, GWh": b["1.2"],
        "Ydinvoima, GWh": b["1.4"],
        "Teollisuus ja rakentaminen, GWh": sec["1"],
        "Palvelut ja julkinen kulutus, GWh": sec["3"],
        "Kotitaloudet ja maatalous, GWh": sec["2"],
    })


def average_price(consumer: str = "I") -> pd.Series:
    """Recent average electricity price for a large business customer.

    Class I = 2,000-19,999 MWh/a.  Class K (70-150 GWh/a) is closest to a data
    centre but is reported less consistently, so both are returned.
    """
    p = pd.read_csv(os.path.join(PROC, "electricity_prices.csv"))
    p = p[(p.contentscode == "hinta_snt_kwh") & (p.timeperiod_m >= "2024M01")]
    out = {}
    for cust in ("I", "J", "K"):
        for comp, lab in (("A", "energia"), ("B", "verkko"), ("SSS", "kokonais")):
            v = p[(p.energia_49_20220905 == cust) & (p.energia_48_20220905 == comp)]["value"]
            if len(v.dropna()):
                out[f"{cust}: {lab}, snt/kWh"] = float(v.dropna().mean())
    return pd.Series(out)


def supply_demand() -> pd.Series:
    ops = m3.operations()
    demand_twh = float(ops["Sähkönkulutus, TWh/vuosi"])
    load_mw = demand_twh * 1e6 / 8760            # average MW

    wind_twh = sc.WIND_MW_NEW * WIND_CF * 8760 / 1e6
    loviisa_twh = LOVIISA_MW * LOVIISA_PPA_SHARE * NUCLEAR_CF * 8760 / 1e6

    # Low-wind hours: what fraction of the load the contracted wind covers.
    wind_low_mw = sc.WIND_MW_NEW * LOW_WIND_OUTPUT_FRACTION
    residual_low_mw = max(load_mw - wind_low_mw, 0.0)

    return pd.Series({
        "Datakeskusten sähkönkulutus, TWh/vuosi": demand_twh,
        "  keskiteho, MW": load_mw,
        "  osuus Suomen kulutuksesta 2024, %": demand_twh / 83.053 * 100,
        "Uusi sopimustuulivoima, MW": sc.WIND_MW_NEW,
        "  tuotanto, TWh/vuosi (CF %.2f)" % WIND_CF: wind_twh,
        "Loviisan PPA-osuus, TWh/vuosi": loviisa_twh,
        "  (huom: olemassa olevaa kapasiteettia, ei uutta)": np.nan,
        "Vuositason jäännöskysyntä (kulutus - uusi tuuli), TWh": demand_twh - wind_twh,
        "Tuulen tuotanto heikkotuulisina tunteina, MW": wind_low_mw,
        "Jäännöskysyntä heikkotuulisina tunteina, MW": residual_low_mw,
        "  osuus datakeskusten kuormasta, %": residual_low_mw / load_mw * 100,
        "Akku, MW": sc.BATTERY_MW,
        "  akun kesto täydellä teholla, h (2h-järjestelmä)": 2.0,
        "  akun kattama osuus kuormasta, %": sc.BATTERY_MW / load_mw * 100,
    })


def price_impact() -> pd.DataFrame:
    """Residual-demand price effect and its distributional decomposition.

        Δp/p = (ΔD/Q) / ε_S

    Reported for the annual balance (new wind counted) and for the low-wind
    hours (wind mostly absent).  The consumer cost is a transfer to producers,
    not a welfare loss; the welfare loss is the small triangle on top.
    """
    bal = electricity_balance()
    Q = bal["Sähkön kokonaiskulutus 2024, GWh"] / 1000        # TWh
    sd = supply_demand()
    prices = average_price()
    p_energy = prices.get("I: energia, snt/kWh", 5.0) * 10     # EUR/MWh

    rows = []
    for label, dD in (("Vuositaso, uusi tuulivoima mukana",
                       sd["Vuositason jäännöskysyntä (kulutus - uusi tuuli), TWh"]),
                      ("Vuositaso, ilman uutta tuulivoimaa",
                       sd["Datakeskusten sähkönkulutus, TWh/vuosi"])):
        for eps in SUPPLY_ELASTICITIES:
            rel = (dD / Q) / eps
            dp = rel * p_energy
            transfer = Q * 1e6 * dp / 1e6          # M EUR: TWh -> MWh x EUR/MWh
            dwl = 0.5 * abs(dD) * 1e6 * dp / 1e6   # rough triangle, M EUR
            rows.append({
                "Skenaario": label,
                "Tarjonnan jousto ε_S": eps,
                "ΔD, TWh": dD,
                "Hintamuutos, %": rel * 100,
                "Hintamuutos, €/MWh": dp,
                "Siirto kuluttajilta tuottajille, M€/vuosi": transfer,
                "Hyvinvointitappio (arvio), M€/vuosi": dwl,
            })
    return pd.DataFrame(rows), pd.Series({
        "Sähkön kokonaiskulutus, TWh": Q,
        "Sähköenergian hinta (yritysasiakas I, 2024-2026), €/MWh": p_energy,
    })


def waste_heat() -> pd.Series:
    """Waste-heat recovery: real, but bounded by the local heat network.

    Google already feeds Hamina's district heating network (about 2,000
    households from autumn 2025).  The physical heat available is close to the
    electricity consumed; what can be sold is limited by how much heat demand
    exists within pipe distance, and Kajaani, Muhos and Vaala are small.
    """
    ops = m3.operations()
    twh = float(ops["Sähkönkulutus, TWh/vuosi"])
    heat_available = twh * 0.95
    dh_price = 85.0            # EUR/MWh, average Finnish district heat
    rows = {}
    for share in (0.05, 0.15, 0.30):
        sold = heat_available * share
        rows[f"Talteenotto {share:.0%}"] = sold
    s = pd.Series(rows)
    out = pd.Series({
        "Hukkalämpöä teoriassa saatavilla, TWh/vuosi": heat_available,
        "Suomen kaukolämmön kulutus, TWh/vuosi (n.)": 33.0,
    })
    for k, v in s.items():
        out[f"{k}: TWh"] = v
        out[f"{k}: arvo kaukolämpönä, M€/vuosi"] = v * 1e6 * dh_price / 1e6
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fmt = lambda x: f"{x:,.2f}"    # noqa: E731

    print("=" * 96)
    print("LÄHESTYMISTAPA 4: SÄHKÖMARKKINAT")
    print("=" * 96)

    print("\n--- 4.1 Suomen sähkötase 2024 (Tilastokeskus) ---")
    bal = electricity_balance()
    for k, v in bal.items():
        print(f"  {k:48s} {v:12,.0f}")
    bal.to_csv(os.path.join(OUT, "m4_balance.csv"))

    print("\n--- 4.2 Sähkön hinta (Tilastokeskus, keskiarvo 2024M01-2026M03) ---")
    pr = average_price()
    for k, v in pr.items():
        print(f"  {k:48s} {v:8,.2f}")
    pr.to_csv(os.path.join(OUT, "m4_prices.csv"))

    print("\n--- 4.3 Kysyntä vs. sopimuksin hankittu tarjonta ---")
    sd = supply_demand()
    for k, v in sd.items():
        print(f"  {k:56s} {'' if pd.isna(v) else f'{v:12,.2f}'}")
    sd.to_csv(os.path.join(OUT, "m4_supply_demand.csv"))

    print("\n--- 4.4 Hintavaikutus ja sen jakautuminen ---")
    pi, meta = price_impact()
    for k, v in meta.items():
        print(f"  {k:56s} {v:10,.2f}")
    print()
    print(pi.to_string(index=False, float_format=fmt))
    pi.to_csv(os.path.join(OUT, "m4_price_impact.csv"), index=False)

    print("\n--- 4.5 Hukkalämmön talteenotto ---")
    wh = waste_heat()
    for k, v in wh.items():
        print(f"  {k:48s} {v:10,.2f}")
    wh.to_csv(os.path.join(OUT, "m4_waste_heat.csv"))


if __name__ == "__main__":
    main()
