"""Minimal client for Statistics Finland's PxWeb (StatFin) API.

The API is documented at https://pxdata.stat.fi/PxWeb/api/v1/  A GET on a table
path returns metadata (variables + allowed values); a POST with a JSON query
returns the data.  We always ask for `json-stat2`, which gives us dimension
sizes and a flat value array we can reshape.

All responses are cached on disk under data/raw so that re-running the analysis
does not hammer the API and so results are reproducible offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
UA = "datakeskus-analyysi/1.0 (economic research; python-urllib)"


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ".json")


def _request(url: str, payload: dict | None = None, retries: int = 4) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"StatFin request failed after {retries} tries: {url}: {last}")


def metadata(table: str, cache_key: str | None = None) -> dict:
    """Fetch a table's variable metadata.  `table` is e.g. 'pt/14yn.px'."""
    key = cache_key or "meta_" + table.replace("/", "_").replace(".px", "")
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    meta = _request(f"{BASE}/{table}")
    with open(path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def query(table: str, selection: dict[str, list[str]], cache_key: str | None = None) -> dict:
    """POST a json-stat2 query.  `selection` maps variable code -> list of values.

    A value list of ["*"] selects everything along that dimension.
    """
    payload = {
        "query": [
            {"code": code,
             "selection": {"filter": "all" if vals == ["*"] else "item",
                           "values": ["*"] if vals == ["*"] else vals}}
            for code, vals in selection.items()
        ],
        "response": {"format": "json-stat2"},
    }
    key = cache_key or "data_" + table.replace("/", "_").replace(".px", "") + "_" + \
        hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    res = _request(f"{BASE}/{table}", payload)
    with open(path, "w") as f:
        json.dump(res, f, ensure_ascii=False)
    return res


def to_frame(js: dict):
    """Flatten a json-stat2 response into a tidy pandas DataFrame.

    One column per dimension (holding the *codes*), plus `label_<dim>` for the
    human-readable text and `value` for the observation.
    """
    import itertools
    import pandas as pd

    dims = js["id"]
    sizes = [js["size"][i] for i in range(len(dims))]
    cats = []
    for d in dims:
        idx = js["dimension"][d]["category"]["index"]
        labels = js["dimension"][d]["category"].get("label", {})
        if isinstance(idx, dict):
            codes = sorted(idx, key=lambda k: idx[k])
        else:
            codes = list(idx)
        cats.append([(c, labels.get(c, c)) for c in codes])

    values = js["value"]
    if isinstance(values, dict):  # sparse form
        n = 1
        for s in sizes:
            n *= s
        dense = [None] * n
        for k, v in values.items():
            dense[int(k)] = v
        values = dense

    rows = []
    for pos, combo in enumerate(itertools.product(*cats)):
        rec = {}
        for d, (code, label) in zip(dims, combo):
            rec[d] = code
            rec["label_" + d] = label
        rec["value"] = values[pos]
        rows.append(rec)
    return pd.DataFrame(rows)
