"""Run the whole analysis end to end.

    python3 src/fetch_data.py      # once, populates data/
    python3 src/run_all.py         # every block, then the charts
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

BLOCKS = [
    ("m1_io_analysis", "Panos-tuotosanalyysi"),
    ("m2_keynesian", "Keynesiläinen kysyntäanalyysi ja tarjontarajoitteet"),
    ("m3_growth", "Tarjontapuoli, pääomakanta ja BKT vs. kansantulo"),
    ("m4_electricity", "Sähkömarkkinat"),
    ("m5_fiscal", "Julkinen talous"),
    ("m6_synthesis", "Yhteenveto"),
]

if __name__ == "__main__":
    import importlib
    for mod, label in BLOCKS:
        m = importlib.import_module(mod)
        m.main()
        print()
    import make_figures
    make_figures.main()
