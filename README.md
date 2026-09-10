# Datakeskusinvestoinnin kansantaloudelliset vaikutukset

Taloustieteellinen analyysi suuren tekoälydatakeskusinvestoinnin vaikutuksista
Suomen kansantalouteen. Esimerkkitapauksena Googlen 9.9.2026 ilmoittama
**13 miljardin euron investointi** Haminaan, Kajaaniin, Muhokselle ja Vaalaan
vuosina 2027–2028.

**➜ [Lue raportti: `raportti.md`](raportti.md)**

## Keskeiset tulokset

| | Rakennusvaihe 2027–2028 | Käyttövaihe (vakaa tila) |
|---|---|---|
| Ilmoitettu investointi | 6 500 M€/v (10,5 % Suomen investoinneista) | 1 607 M€/v korvausinvestointeja |
| Kotimaisuusaste | **19,9 %** | — |
| BKT-vaikutus | 1 188 M€/v (**0,42 %**) | 2 389 M€/v (**0,85 %**) |
| Nettokansantulo | 974 M€/v | **386 M€/v (0,14 %)** |
| Työllisyys | 27 431 htv | 3 699 henkeä pysyvästi |
| Verotuotto | 1 134–1 525 M€ | 239 M€/v |

Kolme havaintoa nousevat yli muiden:

1. **Noin 80 % investoinnista menee suoraan tuontiin.** Tekoälydatakeskus on
   pääosin palvelimia ja kiihdyttimiä, joita Suomessa ei valmisteta.
2. **Käyttövaiheessa BKT nousee 0,85 % mutta nettokansantulo vain 0,14 %.**
   77 % BKT-vaikutuksesta on ulkomaisessa omistuksessa olevien laitteiden
   poistoja, jotka eivät ole kenenkään tuloa Suomessa. Tulos on robusti:
   BKT-vaikutus vaihtelee oletuksista riippuen 2 027–3 022 M€, kansantulo
   pysyy 345–419 M€:ssa.
3. **Sähköveroluokan muutos 1.7.2026 tuottaa valtiolle enemmän (58,5 M€/v)
   kuin hankkeen yhteisövero ja kiinteistövero yhteensä.**

## Menetelmät

Viisi toisistaan riippumatonta lähestymistapaa:

| # | Lähestymistapa | Menetelmä |
|---|---|---|
| 1 | Panos-tuotosanalyysi | Leontief-malli, Type I ja Type II -kertoimet, 64 toimialaa |
| 2 | Keynesiläinen kysyntäanalyysi | Rakenteellinen kerroin + palkkareaktion OLS-estimointi (HAC) |
| 3 | Uusklassinen kasvulaskenta | Tuotantofunktio + BKT→BKTL→NNI-kaskadi |
| 4 | Sähkömarkkina-analyysi | Jäännöskysyntä, tarjonnan jousto, tunti- vs. vuositase |
| 5 | Julkisen talouden analyysi | Efektiiviset veroasteet kansantalouden tilinpidosta |

Kaikki parametrit luetaan Tilastokeskuksen aineistosta tai perustellaan
lähteellä. Kriittiset oletukset esitetään vaihteluväleinä.

## Käyttö

```bash
python3 -m pip install numpy pandas
python3 src/fetch_data.py          # rakentaa taulut hakemistoon data/processed
python3 src/run_all.py             # ajaa kaikki lohkot ja tuottaa kuviot
```

Rajapintavastaukset ovat välimuistissa hakemistossa `data/raw`, joten
`fetch_data.py` toimii myös ilman verkkoyhteyttä ja tuottaa täsmälleen ne
luvut, joihin raportti perustuu. Verkkoyhteydellä ja `--force`-valitsimella
se hakee tuoreimmat tiedot.

Yksittäinen lohko: `python3 src/m3_growth.py`.
Aineiston uudelleenhaku: `python3 src/fetch_data.py --force [taulun_nimi]`.

## Rakenne

```
src/
  statfin.py          Tilastokeskuksen PxWeb-rajapinnan asiakas (levyvälimuisti)
  fetch_data.py       Aineistojen haku ja siivous
  io_model.py         Leontief-malli, Type II -sulkeuma kalibroituna tilinpidosta
  scenario.py         Investointimenojen kohdentaminen kotimaiseen loppukysyntään
  econometrics.py     OLS Newey–West-keskivirheillä
  m1_io_analysis.py   Lähestymistapa 1: panos-tuotos
  m2_keynesian.py     Lähestymistapa 2: kysyntä ja tarjontarajoitteet
  m3_growth.py        Lähestymistapa 3: pääomakanta, BKT vs. kansantulo
  m4_electricity.py   Lähestymistapa 4: sähkömarkkinat
  m5_fiscal.py        Lähestymistapa 5: julkinen talous
  m6_synthesis.py     Yhteenveto ja julkisten väitteiden testaus
  figures.py          Teemaa mukautuva SVG-kuviogeneraattori
  make_figures.py     Raportin kuviot
data/
  raw/                PxWeb-vastaukset välimuistissa (json-stat2), versioidaan
  processed/          Siivotut CSV-taulut (johdetaan raw-hakemistosta)
output/
  tables/             Tulostaulukot CSV-muodossa
  figures/            Kuviot SVG-muodossa
  tulokset.txt        Täydellinen laskenta-ajo
raportti.md           Raportti
```

## Aineistolähteet

Tilastokeskuksen StatFin-tietokanta (PxWeb-rajapinta): panos-tuotostaulukot,
kansantalouden tilinpito, työvoimatutkimus, ansiotasoindeksi, palkkarakenne,
energian hankinta ja kulutus, energian hinnat, rakennuskustannusindeksi.
Täydellinen luettelo raportin luvussa 13.
