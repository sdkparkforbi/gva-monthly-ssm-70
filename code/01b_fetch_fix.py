# -*- coding: utf-8 -*-
"""01 보완: 서비스생산지수(시간분할)·분기부가가치(Q형식)·GPR/EPU 재수집."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import ecos, kosis, to_sql
from importlib import import_module
m01 = import_module("01_fetch_raw") if False else None  # noqa
YM = lambda t: f"{t[:4]}-{t[4:6]}-01"

BOK36 = __import__("01_fetch_raw", fromlist=["BOK36"]).BOK36 if False else None


def fetch_svc_prod_chunked():
    """서비스업생산지수(불변 T2). 4년 창으로 분할 수집."""
    frames = []
    for y0 in range(2000, 2027, 4):
        d = kosis("101", "DT_1KC2020", itmId="T2", objL1="ALL",
                  start=f"{y0}01", end=f"{min(y0+3,2026)}12")
        if len(d):
            frames.append(d)
    if not frames:
        print("  [WARN] svc still empty"); return
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "date": df["PRD_DE"].map(YM),
        "ksic": df["C1"],
        "name": df["C1_NM"].str.strip(),
        "value": pd.to_numeric(df["DT"], errors="coerce"),
    }).dropna(subset=["value"]).drop_duplicates(["date", "ksic"])
    print("  raw_svc_prod:", len(out), "rows | KSIC", out["ksic"].nunique(),
          "| span", out["date"].min(), out["date"].max())
    to_sql(out, "raw_svc_prod")


def fetch_va_quarterly():
    """36산업 분기 실질(200Y104)/명목(200Y103) 부가가치 (Q 형식)."""
    import importlib
    BOK36 = importlib.import_module("01_fetch_raw").BOK36
    recs = []
    for table, kind in [("200Y104", "real"), ("200Y103", "nominal")]:
        for code, name in BOK36.items():
            d = ecos(table, code, cycle="Q", start="1960Q1", end="2026Q4")
            for t, v in zip(d["time"], d["value"]):
                recs.append((t, code, name, kind, v))
    out = pd.DataFrame(recs, columns=["quarter", "code", "name", "kind", "value"])
    print("  raw_va_q:", len(out), "rows | 산업", out["code"].nunique(),
          "| span", out["quarter"].min(), out["quarter"].max())
    to_sql(out, "raw_va_q")


def fetch_gpr_epu():
    recs = []
    try:
        x = pd.read_excel("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls")
        cols = {c.upper(): c for c in x.columns}
        dcol = cols.get("MONTH") or cols.get("DATE") or list(x.columns)[0]
        vcol = cols.get("GPR") or [c for c in x.columns if str(c).upper().startswith("GPR")][0]
        x["d"] = pd.to_datetime(x[dcol], errors="coerce")
        x = x.dropna(subset=["d"])
        xm = x.set_index("d")[vcol].resample("MS").mean().dropna()
        for t, v in xm.items():
            recs.append((t.strftime("%Y-%m-01"), "gpr", float(v)))
        print("  GPR rows", len(xm))
    except Exception as e:
        print("  [WARN] GPR:", str(e)[:90])
    try:
        x = pd.read_excel("https://www.policyuncertainty.com/media/Korea_Policy_Uncertainty_Data.xlsx")
        x = x.dropna(how="all")
        low = {str(c).lower(): c for c in x.columns}
        yc = next(c for k, c in low.items() if k.startswith("year"))
        mc = next(c for k, c in low.items() if k.startswith("month"))
        vc = [c for c in x.columns if "uncertain" in str(c).lower() or "epu" in str(c).lower()]
        vc = vc[-1] if vc else x.columns[-1]
        for _, r in x.iterrows():
            try:
                recs.append((f"{int(r[yc])}-{int(r[mc]):02d}-01", "epu", float(r[vc])))
            except Exception:
                continue
        print("  EPU rows added")
    except Exception as e:
        print("  [WARN] EPU:", str(e)[:90])
    if recs:
        out = pd.DataFrame(recs, columns=["date", "var", "value"]).dropna()
        print("  raw_gpr_epu:", len(out), "| vars", sorted(out["var"].unique()))
        to_sql(out, "raw_gpr_epu")


if __name__ == "__main__":
    print("svc"); fetch_svc_prod_chunked()
    print("va_q"); fetch_va_quarterly()
    print("gpr/epu"); fetch_gpr_epu()
    print("DONE")
