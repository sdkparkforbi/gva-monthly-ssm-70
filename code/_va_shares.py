# -*- coding: utf-8 -*-
"""횡단면 부가가치 비중용 KSIC 중분류 부가가치 입수(기준연도, 공개통계).

  제조·광업: KOSIS 광업제조업조사 DT_1FJY1004_S (T1=부가가치, 합계행 C2=C3=C4='0')  → KSIC(C/B)
  서비스   : KOSIS 서비스업조사   DT_1I303015  (T7=부가가치, 합계 C2='000'·C3='0100') → KSIC(숫자)
저장: data/va_mfg.csv (ksic, va), data/va_svc.csv (ksic_num, va)
"""
import sys
import pandas as pd
sys.path.insert(0, ".")
from _common import kosis, DATA
import os

B = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return float("nan")


def fetch_mfg():
    df = kosis("101", "DT_1FJY1004_S", itmId="T1", objL1="ALL", objL2="ALL",
               objL3="ALL", objL4="ALL", prdSe="Y", start="2022", end="2022")
    if df.empty:
        df = kosis("101", "DT_1FJY1004_S", itmId="T1", objL1="ALL", objL2="ALL",
                   objL3="ALL", objL4="ALL", prdSe="Y", start="2020", end="2023")
    tot = df[(df["C2"] == "0") & (df["C3"] == "0") & (df["C4"] == "0")].copy()
    tot["va"] = tot["DT"].map(num)
    out = tot[["C1", "va"]].dropna().rename(columns={"C1": "ksic"})
    out = out[out["ksic"].str.match(r"^[BC]\d\d$")]   # KSIC 2자리(B/C 중분류)
    out.to_csv(os.path.join(DATA, "va_mfg.csv"), index=False, encoding="utf-8-sig")
    return out


def fetch_svc():
    frames = []
    for y in ["2024", "2023", "2022"]:
        df = kosis("101", "DT_1I303015", itmId="T7", objL1="ALL", objL2="ALL",
                   objL3="ALL", prdSe="Y", start=y, end=y)
        if len(df):
            frames.append(df); break
    df = pd.concat(frames) if frames else pd.DataFrame()
    tot = df[(df["C2"] == "000") & (df["C3"] == "0100")].copy()
    tot["va"] = tot["DT"].map(num)
    tot = tot[tot["C1"].str.match(r"^\d\d$")]          # 2자리 숫자(중분류)
    out = tot[["C1", "va"]].dropna().rename(columns={"C1": "ksic_num"})
    out.to_csv(os.path.join(DATA, "va_svc.csv"), index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    m = fetch_mfg(); s = fetch_svc()
    print("va_mfg:", len(m), "중분류 | 예:", m.head(4).to_dict("records"))
    print("va_svc:", len(s), "중분류 | 예:", s.head(4).to_dict("records"))
