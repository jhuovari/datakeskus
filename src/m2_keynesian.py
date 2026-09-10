"""Approach 2 -- Keynesian demand analysis with supply constraints.

The input-output block of Approach 1 answers "how much Finnish production does
this demand call for, if the resources are available".  This block asks the
question the Leontief model cannot: **are they available, and what happens to
the part that is not?**

Three pieces:
  1. a structural open-economy multiplier, derived from national-accounts
     leakages, as an independent cross-check on the IO Type II figure;
  2. slack diagnostics -- how much idle labour and construction capacity the
     Finnish economy actually has in 2026, nationally and regionally;
  3. an econometric estimate of how construction wages respond to construction
     activity, which prices the crowding-out channel instead of asserting it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from econometrics import OLS           # noqa: E402
from io_model import IOSystem         # noqa: E402
import scenario as sc                 # noqa: E402
import m1_io_analysis as m1           # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "tables")
GDP_2025 = m1.GDP_2025
N_YEARS = m1.N_YEARS

# Host municipalities and the regions they sit in.
HOST_REGIONS = {"MK08": "Kymenlaakso (Hamina)",
                "MK18": "Kainuu (Kajaani, Vaala)",
                "MK17": "Pohjois-Pohjanmaa (Muhos)"}


# --------------------------------------------------------------- multiplier
def structural_multiplier(year: int = 2023) -> pd.Series:
    """Textbook open-economy multiplier, with every leakage read off the data.

        k = 1 / (1 - c_d)   where   c_d = APC x (1 - direct tax wedge)
                                          x domestic content of consumption

    This is deliberately the crudest possible model.  Its value is that it uses
    no input-output information at all, so agreement with the Type II figure is
    informative rather than mechanical.
    """
    s = IOSystem.load(year)
    p = s.household_params
    # Share of a euro of extra domestic production that returns as domestic
    # consumption demand: labour share of value added x net-of-wedge x APC x
    # domestic content.
    labour_share = s.comp.sum() / s.va.sum()
    c_d = labour_share * p["wage_to_net_income"] * p["apc"] * p["consumption_domestic_share"]
    k = 1.0 / (1.0 - c_d)
    va_share = s.va.sum() / s.x.sum()
    return pd.Series({
        "Palkkojen osuus arvonlisäyksestä": labour_share,
        "Palkoista kotitalouksien nettotuloon": p["wage_to_net_income"],
        "Keskimääräinen kulutusaste (APC)": p["apc"],
        "Kulutuksen kotimaisuusaste": p["consumption_domestic_share"],
        "Kotimaiseen kysyntään palaava osuus, c_d": c_d,
        "Kerroin k = 1/(1-c_d)": k,
        "Arvonlisäyksen osuus tuotoksesta": va_share,
    })


def crosscheck(year: int = 2023) -> pd.DataFrame:
    """Compare the naive multiplier with the disaggregated IO result."""
    s = IOSystem.load(year)
    shock = sc.build_shock(s.industries)
    sm = structural_multiplier(year)
    t1 = s.impact(shock.f)
    t2 = s.impact(shock.f, type2=True)

    # Naive: value added from the first round, then scaled by the multiplier.
    naive_va = shock.f.sum() * sm["Arvonlisäyksen osuus tuotoksesta"] * sm["Kerroin k = 1/(1-c_d)"]
    rows = {
        "Yksinkertainen kerroinmalli (ei panos-tuotosta)": naive_va,
        "Panos-tuotos, Type I (suora + välillinen)": t1["va_total"],
        "Panos-tuotos, Type II (+ tulokerrannais)": t2["va_total"],
    }
    df = pd.DataFrame({"Arvonlisäys, M€ (koko ohjelma)": rows})
    df["M€ / vuosi"] = df.iloc[:, 0] / N_YEARS
    df["% BKT:sta / vuosi"] = df["M€ / vuosi"] / GDP_2025 * 100
    return df


# ------------------------------------------------------------------- slack
def slack_diagnostics() -> pd.DataFrame:
    lfs = pd.read_csv(os.path.join(PROC, "lfs_annual.csv"))
    lfs = lfs[lfs.ikaryhma_19_20190101 == "15-74"]
    piv = lfs.pivot_table(index="contentscode", columns="timeperiod_y", values="value")

    emp = pd.read_csv(os.path.join(PROC, "na_employment.csv"))
    con = emp[(emp.toimiala_79_20180101 == "F") &
              (emp.taloustoimi_1_20180101 == "E1")].set_index("timeperiod_y")["value"]
    tot = emp[(emp.toimiala_79_20180101 == "SSS") &
              (emp.taloustoimi_1_20180101 == "E1")].set_index("timeperiod_y")["value"]

    peak_year = int(con.loc[2000:2025].idxmax())
    rows = {
        "Työttömyysaste 2019, %": piv.loc["tyti-Tyottomyysaste", 2019],
        "Työttömyysaste 2025, %": piv.loc["tyti-Tyottomyysaste", 2025],
        "Työttömiä 2025, 1000 henkeä": piv.loc["tyti-Tyottomat", 2025],
        "Työttömien lisäys 2019->2025, 1000 henkeä":
            piv.loc["tyti-Tyottomat", 2025] - piv.loc["tyti-Tyottomat", 2019],
        "Työllisyysaste 2025 (15-74), %": piv.loc["tyti-Tyollisyysaste", 2025],
        f"Rakentamisen työlliset, huippu {peak_year}, 1000": con.loc[peak_year],
        "Rakentamisen työlliset 2025, 1000": con.loc[2025],
        "Rakentamisen työllisyysvaje huipusta, 1000": con.loc[peak_year] - con.loc[2025],
        "Rakentamisen työllisyysvaje 2019-tasosta, 1000": con.loc[2019] - con.loc[2025],
        "Rakentamisen osuus työllisistä 2025, %": con.loc[2025] / tot.loc[2025] * 100,
    }
    return pd.Series(rows).to_frame("Arvo")


def regional_slack() -> pd.DataFrame:
    r = pd.read_csv(os.path.join(PROC, "lfs_region.csv"))
    r = r[r.timeperiod_y == 2025]
    piv = r.pivot_table(index=["alue_23_20180101", "label_alue_23_20180101"],
                        columns="contentscode", values="value")
    keep = ["tyti-Vaesto", "tyti-Tyovoima", "tyti-Tyolliset", "tyti-Tyottomat",
            "tyti-Tyottomyysaste", "Tyollisyysaste_15_64"]
    piv = piv[keep]
    piv.columns = ["Väestö 15-74, 1000", "Työvoima, 1000", "Työlliset, 1000",
                   "Työttömät, 1000", "Työttömyysaste, %", "Työllisyysaste 15-64, %"]
    sel = piv.loc[piv.index.get_level_values(0).isin(list(HOST_REGIONS) + ["SSS"])]
    return sel.droplevel(0)


# ---------------------------------------------------- wage response (econometrics)
def wage_response() -> tuple[dict, pd.DataFrame]:
    """How much do construction wages accelerate when construction booms?

    Dependent variable: year-on-year growth of the construction earnings index
    *minus* the same growth for the whole economy.  Using the relative wage
    strips out national wage rounds and consumer-price inflation, isolating
    sector-specific pressure.

    Regressors:
      hours_growth        year-on-year growth of hours worked in construction,
                          i.e. the activity measure the investment shifts;
      rel_agreed_growth   the corresponding difference in collectively agreed
                          pay, so the activity coefficient picks up wage drift
                          rather than negotiated rises;
      interaction         hours growth times a labour-market tightness measure
                          (negative of the demeaned national unemployment rate),
                          which tests whether the wage response is stronger when
                          the labour market is tight.  This matters directly:
                          Finland enters 2027 with 9.7% unemployment.

    Statistics Finland publishes the 2015=100 index levels only from 2015Q1, so
    the year-on-year percentage series are used instead, which reach back to
    2011Q1 and roughly halve the standard errors.
    """
    e = pd.read_csv(os.path.join(PROC, "earnings_q.csv"))
    e = e[e.palk_muo_2_20120101 == 0]
    tot = pd.read_csv(os.path.join(PROC, "earnings_total_q.csv"))

    def by_industry(code, content):
        s = e[(e.toimiala_109_20150101 == code) & (e.contentscode == content)]
        return s.set_index("timeperiod_q")["value"].sort_index().dropna()

    def economy(content):
        s = tot[tot.contentscode == content]
        return s.set_index("timeperiod_q")["value"].sort_index().dropna()

    lfs = pd.read_csv(os.path.join(PROC, "lfs_industry_q.csv"))
    hrs = lfs[(lfs.toimiala_79_20180101 == "F") & (lfs.contentscode == "tyotunnit")]
    hrs = hrs.set_index("timeperiod_q")["value"].sort_index()

    # National unemployment rate, quarterly, as the tightness proxy.  Annual
    # LFS data are interpolated onto quarters -- we only need a slow-moving
    # cyclical state variable, not a high-frequency series.
    lfa = pd.read_csv(os.path.join(PROC, "lfs_annual.csv"))
    u_ann = (lfa[(lfa.ikaryhma_19_20190101 == "15-74") &
                 (lfa.contentscode == "tyti-Tyottomyysaste")]
             .set_index("timeperiod_y")["value"].sort_index())

    d = pd.DataFrame({
        "rel_wage_growth": by_industry("F", "ati_vuosimuutosprosentti")
        - economy("ati_vuosimuutosprosentti"),
        "rel_agreed_growth": by_industry("F", "spi_vuosimuutosprosentti")
        - economy("spi_vuosimuutosprosentti"),
        "hours_growth": np.log(hrs).diff(4) * 100,
    }).dropna()
    d["year"] = d.index.str.slice(0, 4).astype(int)
    d["u_rate"] = d.year.map(u_ann)
    d = d.dropna()
    # Tightness: high when unemployment is low.  Demeaned so the level
    # coefficient stays interpretable as the average-cycle response.
    d["tightness"] = -(d.u_rate - d.u_rate.mean())
    d["interaction"] = d.hours_growth * d.tightness

    base = OLS(d.rel_wage_growth.to_numpy(),
               d[["hours_growth", "rel_agreed_growth"]].to_numpy(),
               names=["rakentamisen tuntien kasvu, %", "sopimuspalkkaero, %-p"],
               hac_lags=4)
    inter = OLS(d.rel_wage_growth.to_numpy(),
                d[["hours_growth", "rel_agreed_growth", "tightness", "interaction"]].to_numpy(),
                names=["rakentamisen tuntien kasvu, %", "sopimuspalkkaero, %-p",
                       "kireys (-työttömyysaste, demeanattu)",
                       "tunnit x kireys"],
                hac_lags=4)
    return {"perus": base, "tilariippuva": inter}, d


def crowding_out(models: dict) -> pd.DataFrame:
    """Price and volume crowding-out under alternative displacement rates.

    Displacement rate d: the share of the demand injection that is met by
    pulling resources out of other Finnish activity rather than out of idle
    capacity.  We report the net GDP effect for a range of d, and separately
    the construction wage pressure the activity increase implies given the
    estimated elasticity.
    """
    s = IOSystem.load(2023)
    shock = sc.build_shock(s.industries)
    t2 = s.impact(shock.f, type2=True)
    x2 = m1.type2_output(s, shock.f)
    iF = s.idx("F")

    # How large is the construction shock relative to the sector?
    constr_output_per_year = x2[iF] / N_YEARS
    constr_base = s.x[iF]
    activity_pct = constr_output_per_year / constr_base * 100

    base, inter = models["perus"], models["tilariippuva"]
    beta = base.beta[1]           # % relative wage growth per % hours growth
    wage_pressure = beta * activity_pct

    # State-dependent response, evaluated at the 2025 unemployment rate.  The
    # interaction is on demeaned tightness, so we need the 2025 deviation.
    lfa = pd.read_csv(os.path.join(PROC, "lfs_annual.csv"))
    u = (lfa[(lfa.ikaryhma_19_20190101 == "15-74") &
             (lfa.contentscode == "tyti-Tyottomyysaste")]
         .set_index("timeperiod_y")["value"].sort_index())
    u_sample = u.loc[2011:2025]
    tight_2025 = -(u.loc[2025] - u_sample.mean())
    beta_2025 = inter.beta[1] + inter.beta[4] * tight_2025
    wage_pressure_2025 = beta_2025 * activity_pct

    rows = []
    for d in (0.0, 0.15, 0.30, 0.50, 0.70):
        va = t2["va_total"] * (1 - d)
        rows.append({
            "Syrjäytymisaste d": d,
            "Nettovaikutus arvonlisäykseen, M€": va,
            "M€ / vuosi": va / N_YEARS,
            "% BKT:sta / vuosi": va / N_YEARS / GDP_2025 * 100,
            "Työllisyys, htv": t2["emp_total"] * (1 - d),
        })
    df = pd.DataFrame(rows)
    meta = pd.Series({
        "Rakentamisen tuotoslisäys, M€/vuosi": constr_output_per_year,
        "Rakentamisen tuotos 2023, M€": constr_base,
        "Aktiviteetin lisäys, % toimialan tuotoksesta": activity_pct,
        "Estimoitu beta, keskimääräinen suhdanne": beta,
        "Implikoitu suhteellinen palkkapaine, %-p (keskim.)": wage_pressure,
        "Työmarkkinoiden kireys 2025 (poikkeama keskiarvosta, %-p)": tight_2025,
        "Estimoitu beta vuoden 2025 tilanteessa": beta_2025,
        "Implikoitu palkkapaine 2025 tilanteessa, %-p": wage_pressure_2025,
        "Rakentamisen työvoimatarve, htv/vuosi (Type II, toimiala F)":
            s.emp_coef[iF] * x2[iF] / N_YEARS,
    })
    return df, meta


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fmt = lambda x: f"{x:,.3f}"      # noqa: E731

    print("=" * 96)
    print("LÄHESTYMISTAPA 2: KEYNESILÄINEN KYSYNTÄANALYYSI JA TARJONTARAJOITTEET")
    print("=" * 96)

    print("\n--- 2.1 Rakenteellinen kerroin kansantalouden tilinpidosta ---")
    sm = structural_multiplier()
    for k, v in sm.items():
        print(f"  {k:48s} {v:8.4f}")

    print("\n--- 2.2 Ristiintarkistus: kerroinmalli vs. panos-tuotos ---")
    cc = crosscheck()
    print(cc.to_string(float_format=lambda x: f"{x:,.1f}"))
    cc.to_csv(os.path.join(OUT, "m2_crosscheck.csv"))

    print("\n--- 2.3 Talouden vapaa kapasiteetti (koko maa) ---")
    sd = slack_diagnostics()
    print(sd.to_string(float_format=lambda x: f"{x:,.1f}"))
    sd.to_csv(os.path.join(OUT, "m2_slack.csv"))

    print("\n--- 2.4 Alueellinen työvoimatilanne 2025 (isäntämaakunnat) ---")
    rs = regional_slack()
    print(rs.to_string(float_format=lambda x: f"{x:,.1f}"))
    rs.to_csv(os.path.join(OUT, "m2_regional.csv"))

    print("\n--- 2.5 Rakentamisen palkkareaktio (OLS, HAC-keskivirheet) ---")
    models, d = wage_response()
    print("  Selitettävä: rakentamisen ansiotasoindeksin vuosimuutos "
          "miinus koko talouden vuosimuutos, %-p")
    print(f"  Otos: {d.index.min()} - {d.index.max()}")
    for nm, mod in models.items():
        print(f"\n  [{nm}]")
        print(mod.summary())
    d.to_csv(os.path.join(OUT, "m2_wage_regression_data.csv"))

    print("\n--- 2.6 Syrjäytymisvaikutus ja hintapaine ---")
    co, meta = crowding_out(models)
    for k, v in meta.items():
        print(f"  {k:60s} {v:10,.3f}")
    print()
    print(co.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    co.to_csv(os.path.join(OUT, "m2_crowding_out.csv"), index=False)
    meta.to_csv(os.path.join(OUT, "m2_crowding_meta.csv"))


if __name__ == "__main__":
    main()
