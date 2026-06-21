# -*- coding: utf-8 -*-
"""Phase A — 정제·변환 → clean_* 테이블.

  clean_va_real / clean_va_nom : 36 부모 분기 실질/명목 부가가치(2000Q1~2025Q4)
  clean_stock_real             : 산업별 실질주가가치 = KOSPI섹터 ÷ (CPI/100), 월
  clean_exog_m                 : 월별 외생(정상성 변환) + 참조레벨
  clean_adf                    : 외생 ADF p값
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import read_sql, to_sql, mlog_diff, adf_p, DATA
import os

Q_START, Q_END = "2000Q1", "2025Q4"


def qkey(q):
    y, n = int(q[:4]), int(q[-1]); return y * 4 + n


def build_va():
    df = read_sql("select quarter, code, name, kind, value from raw_va_q")
    df = df[(df["quarter"].map(qkey) >= qkey(Q_START)) &
            (df["quarter"].map(qkey) <= qkey(Q_END))]
    real = df[df["kind"] == "real"][["quarter", "code", "name", "value"]]
    nom = df[df["kind"] == "nominal"][["quarter", "code", "name", "value"]]
    to_sql(real, "clean_va_real"); to_sql(nom, "clean_va_nom")
    print(f"  clean_va_real: {len(real)} | 부모 {real['code'].nunique()} | {real['quarter'].min()}~{real['quarter'].max()}")
    return real


def build_stock_real():
    """실질주가가치 = 명목 KOSPI섹터지수 ÷ (CPI_kr/100)."""
    cpi = read_sql("select date, value from raw_exog_m where var='cpi_kr'").set_index("date")["value"]
    ks = read_sql("select date, sector_code, sector_name, value from raw_kospi_sector")
    ks = ks.merge(cpi.rename("cpi"), left_on="date", right_index=True, how="left")
    ks["real_stock"] = ks["value"] / (ks["cpi"] / 100.0)
    out = ks.dropna(subset=["real_stock"])[["date", "sector_code", "sector_name", "real_stock"]]
    to_sql(out, "clean_stock_real")
    print(f"  clean_stock_real: {len(out)} | 섹터 {out['sector_code'].nunique()} | {out['date'].min()}~{out['date'].max()}")


def build_exog():
    ex = read_sql("select date, var, value from raw_exog_m")
    w = ex.groupby(["date", "var"])["value"].mean().unstack().sort_index()
    gpr = read_sql("select date, value from raw_gpr_epu where var='gpr'")
    if len(gpr):
        w = w.join(gpr.set_index("date")["value"].rename("gpr"), how="left")
    out = pd.DataFrame(index=w.index)
    out["oil_g"] = mlog_diff(w["oil"]) if "oil" in w else np.nan
    trd = (w["exp_vol"] + w["imp_vol"]) / 2 if {"exp_vol", "imp_vol"} <= set(w) else np.nan
    out["trd_g"] = mlog_diff(trd) if trd is not np.nan else np.nan
    cpi_yoy = 100 * (w["cpi_kr"] / w["cpi_kr"].shift(12) - 1)
    rrate = w["rate"] - cpi_yoy
    out["d_rrate"] = rrate.diff()
    rfx = w["fx"] * w["cpi_us"] / w["cpi_kr"]
    out["rfx_g"] = mlog_diff(rfx)
    if "m2" in w:
        out["m2_g"] = mlog_diff(w["m2"])
    if "emp" in w:
        out["lab_g"] = mlog_diff(w["emp"])              # 취업자 증가율(노동공급 채널)
    if "gpr" in w:
        out["gpr_l"] = 100 * np.log(w["gpr"])           # 로그 GPR(준정상)
    # 참조 레벨
    out["rrate_lvl"] = rrate
    out["cpi_kr"] = w["cpi_kr"]
    out = out.reset_index().rename(columns={"index": "date"})
    to_sql(out, "clean_exog_m")
    cov = []
    for c in [x for x in out.columns if x not in ("date",)]:
        s = out[c].dropna()
        cov.append([c, s.index.min(), len(s), adf_p(out[c])])
    cov = pd.DataFrame(cov, columns=["var", "first_row", "n", "adf_p"])
    to_sql(cov, "clean_adf")
    cov.to_csv(os.path.join(DATA, "exog_adf_monthly.csv"), index=False, encoding="utf-8-sig")
    print("  clean_exog_m:", [c for c in out.columns if c != "date"])
    print(cov.to_string(index=False))


if __name__ == "__main__":
    print("1) 분기 부가가치"); build_va()
    print("2) 실질주가가치");  build_stock_real()
    print("3) 월별 외생");     build_exog()
    print("DONE")
