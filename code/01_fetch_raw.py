# -*- coding: utf-8 -*-
"""Phase A — 원자료 수집 → SQLite raw_* 테이블.

소스(모두 KOSIS/한국은행, 유가만 FRED):
  raw_kospi_sector : KOSPI 산업별 주가지수 25섹터        [KOSIS 343 DT_343_2010_S0190]  (헤드라인 실질주가가치 원자료)
  raw_mfg_prod     : 산업별 광공업생산지수(전국, KSIC)    [KOSIS 101 DT_1F02001]
  raw_svc_prod     : 산업별 서비스업생산지수(불변, KSIC)  [KOSIS 101 DT_1KC2020]
  raw_va_q         : 36산업 분기 명목·실질 부가가치        [BOK 200Y103/200Y104]
  raw_exog_m       : 월별 외생(유가·환율·금리·CPI·교역·M2) [FRED + BOK]
  raw_gpr_epu      : 월별 GPR / 한국 EPU(가능시)           [Caldara-Iacoviello / policyuncertainty]
"""
import sys, io
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, ".")
from _common import ecos, ecos_month_series, kosis, fred, to_sql, KEY_KOSIS

YM = lambda t: f"{t[:4]}-{t[4:6]}-01"

# ---- BOK 36산업 코드(실질부가가치 200Y104 / 명목 200Y103) ----
BOK36 = {
 "1101":"농림어업","1102":"광업","110301":"음식료품","110302":"섬유및가죽","110303":"목재종이인쇄",
 "110304":"코크스석유정제","110305":"화학","110306":"비금속광물","110307":"1차금속","110308":"금속가공",
 "110309":"컴퓨터전자광학","110310":"전기장비","110311":"기계장비","110312":"운송장비","110313":"기타제조및수리",
 "110401":"전기업","110402":"가스증기","110403":"수도하수폐기물","11051":"주거용건물건설","11052":"비주거용건물건설",
 "11053":"토목건설","11054":"건축보수","110601":"도소매","110602":"숙박음식","1107":"운수",
 "1108":"금융보험","1109":"부동산","11141":"통신","11142":"출판방송정보","111501":"전문과학기술",
 "111502":"사업지원","1110":"공공행정","1111":"교육","1112":"의료보건복지","11131":"예술스포츠여가","11132":"기타서비스",
}


def fetch_kospi_sector():
    """KOSPI 산업별 주가지수(월, 25섹터, 2000.01~)."""
    df = kosis("343", "DT_343_2010_S0190", itmId="13103130657T1", objL1="ALL",
               start="200001", end="202612")
    if df.empty:
        print("  [WARN] KOSPI sector empty"); return
    out = pd.DataFrame({
        "date": df["PRD_DE"].map(YM),
        "sector_code": df["C1"],
        "sector_name": df["C1_NM"].str.strip(),
        "value": pd.to_numeric(df["DT"], errors="coerce"),
    }).dropna(subset=["value"])
    print(" ", to_sql(out, "raw_kospi_sector"),
          "| 섹터", out["sector_name"].nunique(), "| 기간", out["date"].min(), out["date"].max())


def _kosis_prod_item(org, tbl):
    """생산지수 itm 코드 자동탐지(이름에 '생산' 포함, '계절조정' 우선 제외)."""
    df = kosis(org, tbl, itmId="ALL", objL1="ALL", objL2="ALL",
               start="202601", end="202602")
    if df.empty or "ITM_NM" not in df:
        return None
    items = df[["ITM_ID", "ITM_NM"]].drop_duplicates()
    # '불변'/'생산' 우선, '계절조정' 회피
    cand = items[items["ITM_NM"].str.contains("생산|불변", na=False)]
    cand = cand[~cand["ITM_NM"].str.contains("계절", na=False)]
    if len(cand):
        return cand.iloc[0]["ITM_ID"]
    return items.iloc[0]["ITM_ID"]


def fetch_mfg_prod():
    """산업별 광공업생산지수(전국=C1 00, KSIC=C2), 월."""
    itm = _kosis_prod_item("101", "DT_1F02001") or "13103114447T1"
    df = kosis("101", "DT_1F02001", itmId=itm, objL1="00", objL2="ALL",
               start="200001", end="202612")
    if df.empty:
        df = kosis("101", "DT_1F02001", itmId="ALL", objL1="00", objL2="ALL",
                   start="200001", end="202612")
        df = df[df["ITM_NM"].str.contains("생산", na=False)] if not df.empty else df
    if df.empty:
        print("  [WARN] mfg prod empty"); return
    out = pd.DataFrame({
        "date": df["PRD_DE"].map(YM),
        "ksic": df["C2"],
        "name": df["C2_NM"].str.strip(),
        "value": pd.to_numeric(df["DT"], errors="coerce"),
    }).dropna(subset=["value"])
    print(" ", to_sql(out, "raw_mfg_prod"),
          "| KSIC", out["ksic"].nunique(), "| 기간", out["date"].min(), out["date"].max())


def fetch_svc_prod():
    """산업별 서비스업생산지수(불변지수 T2, KSIC=C1), 월."""
    df = kosis("101", "DT_1KC2020", itmId="T2", objL1="ALL",
               start="200001", end="202612")
    if df.empty:
        print("  [WARN] svc prod empty"); return
    out = pd.DataFrame({
        "date": df["PRD_DE"].map(YM),
        "ksic": df["C1"],
        "name": df["C1_NM"].str.strip(),
        "value": pd.to_numeric(df["DT"], errors="coerce"),
    }).dropna(subset=["value"])
    print(" ", to_sql(out, "raw_svc_prod"),
          "| KSIC", out["ksic"].nunique(), "| 기간", out["date"].min(), out["date"].max())


def fetch_va_quarterly():
    """36산업 분기 실질(200Y104)·명목(200Y103) 부가가치."""
    recs = []
    for table, kind in [("200Y104", "real"), ("200Y103", "nominal")]:
        for code, name in BOK36.items():
            d = ecos(table, code, cycle="Q", start="196001", end="202604")
            if d.empty:
                continue
            for t, v in zip(d["time"], d["value"]):
                recs.append((t, code, name, kind, v))
    out = pd.DataFrame(recs, columns=["quarter", "code", "name", "kind", "value"])
    print(" ", to_sql(out, "raw_va_q"),
          "| 산업", out["code"].nunique(), "| 기간", out["quarter"].min(), out["quarter"].max())


def fetch_exog_monthly():
    """월별 외생 블록(원수준). 변환은 03_clean에서."""
    S = {}
    try:
        S["oil"] = fred("MCOILBRENTEU")                       # Brent USD/bbl
    except Exception as e:
        print("  [WARN] brent:", e)
    S["fx"] = ecos_month_series("731Y006", "0000003")          # 원/달러 종가
    S["rate"] = ecos_month_series("721Y001", "7020000")        # 회사채3년 AA-
    S["cpi_kr"] = ecos_month_series("901Y009", "0")            # 한국 CPI
    S["cpi_us"] = ecos_month_series("902Y008", "US")           # 미국 CPI
    S["exp_vol"] = ecos_month_series("403Y002", "*AA")         # 수출물량지수
    S["imp_vol"] = ecos_month_series("403Y004", "*AA")         # 수입물량지수
    for tbl, it, nm in [("101Y004", "BBHA00", "m2"), ("101Y003", "BBHS00", "m2b")]:
        s = ecos_month_series(tbl, it)
        if len(s):
            S["m2"] = s; break
    recs = []
    for var, s in S.items():
        if s is None or not len(s):
            print(f"  [WARN] exog {var} empty"); continue
        for t, v in s.items():
            recs.append((t.strftime("%Y-%m-01"), var, float(v)))
    out = pd.DataFrame(recs, columns=["date", "var", "value"])
    print(" ", to_sql(out, "raw_exog_m"), "| 변수", sorted(out["var"].unique()))


def fetch_gpr_epu():
    """월별 GPR(Caldara-Iacoviello)·한국 EPU(Baker-Bloom-Davis). 실패시 생략."""
    recs = []
    # GPR
    for url, col in [("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls", "GPR"),
                     ("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", "GPRD")]:
        try:
            x = pd.read_excel(url)
            cols = {c.upper(): c for c in x.columns}
            dcol = cols.get("MONTH") or cols.get("DATE") or list(x.columns)[0]
            vcol = cols.get(col) or [c for c in x.columns if c.upper().startswith("GPR")][0]
            x["d"] = pd.to_datetime(x[dcol], errors="coerce")
            x = x.dropna(subset=["d"])
            xm = x.set_index("d")[vcol].resample("MS").mean().dropna()
            for t, v in xm.items():
                recs.append((t.strftime("%Y-%m-01"), "gpr", float(v)))
            print("  GPR rows", len(xm)); break
        except Exception as e:
            print("  [WARN] GPR:", str(e)[:80])
    # Korea EPU
    for url in ["https://www.policyuncertainty.com/media/Korea_Policy_Uncertainty_Data.xlsx"]:
        try:
            x = pd.read_excel(url)
            x = x.dropna(how="all")
            yc = [c for c in x.columns if str(c).lower().startswith("year")][0]
            mc = [c for c in x.columns if str(c).lower().startswith("month")][0]
            vc = [c for c in x.columns if "uncertain" in str(c).lower() or "index" in str(c).lower()][-1]
            for _, r in x.iterrows():
                try:
                    y = int(r[yc]); m = int(r[mc])
                except Exception:
                    continue
                recs.append((f"{y}-{m:02d}-01", "epu", float(r[vc])))
            print("  EPU rows added")
        except Exception as e:
            print("  [WARN] EPU:", str(e)[:80])
    if recs:
        out = pd.DataFrame(recs, columns=["date", "var", "value"]).dropna()
        print(" ", to_sql(out, "raw_gpr_epu"), "| 변수", sorted(out["var"].unique()))
    else:
        print("  [WARN] GPR/EPU 모두 실패 — 생략")


if __name__ == "__main__":
    print("1) KOSPI 산업별 주가지수");   fetch_kospi_sector()
    print("2) 산업별 광공업생산지수");    fetch_mfg_prod()
    print("3) 산업별 서비스업생산지수");  fetch_svc_prod()
    print("4) 36산업 분기 부가가치");      fetch_va_quarterly()
    print("5) 월별 외생");                fetch_exog_monthly()
    print("6) GPR/EPU");                  fetch_gpr_epu()
    print("DONE.")
