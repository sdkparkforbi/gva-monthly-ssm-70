# -*- coding: utf-8 -*-
"""공통 유틸: 경로·SQLite·BOK ECOS·KOSIS·FRED 수집기.

선행연구(gva-repo)와 분리된 새 코드베이스. 단일 SQLite(db/gva_monthly.sqlite)에
raw_* / clean_* / latent_* / result_* 테이블로 적재한다.
"""
import io
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "gva_monthly.sqlite")
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.dirname(DB), exist_ok=True)
os.makedirs(DATA, exist_ok=True)

KEY_BOK = open(os.path.join(ROOT, "API_KEY_BOK.txt"), encoding="utf-8").read().strip()
KEY_KOSIS = open(os.path.join(ROOT, "API_KEY_KOSIS.txt"), encoding="utf-8").read().strip()


def con():
    return sqlite3.connect(DB)


def to_sql(df, table, if_exists="replace"):
    with con() as c:
        df.to_sql(table, c, if_exists=if_exists, index=False)
    return f"{table}: {len(df)} rows"


def read_sql(q):
    with con() as c:
        return pd.read_sql(q, c)


# ----------------------------------------------------------------- BOK ECOS
def ecos(table, *items, cycle="M", start="190001", end="209912", retry=3):
    """ECOS StatisticSearch → long DataFrame(time, value). 월 'YYYYMM', 분기 'YYYYQn'."""
    it = "/".join(items) if items else "?"
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY_BOK}/json/kr/1/100000/"
           f"{table}/{cycle}/{start}/{end}/{it}")
    for k in range(retry):
        try:
            j = requests.get(url, timeout=120).json()
            rows = j.get("StatisticSearch", {}).get("row", [])
            recs = []
            for r in rows:
                v = r.get("DATA_VALUE")
                if v in (None, "", "-"):
                    continue
                recs.append((r["TIME"], float(v)))
            return pd.DataFrame(recs, columns=["time", "value"])
        except Exception as e:
            if k == retry - 1:
                raise
            time.sleep(2)


def ecos_month_series(table, *items):
    """월 ECOS → pandas Series(index=Timestamp(월초))."""
    df = ecos(table, *items, cycle="M")
    if df.empty:
        return pd.Series(dtype=float)
    idx = [pd.Timestamp(int(t[:4]), int(t[4:6]), 1) for t in df["time"]]
    return pd.Series(df["value"].values, index=idx).sort_index()


# ----------------------------------------------------------------- KOSIS
KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def kosis(org, tbl, itmId="ALL", start="200001", end="202612",
          objL1="ALL", objL2="", objL3="", objL4="", prdSe="M", retry=3):
    """KOSIS 파라미터 데이터 → long DataFrame. 빈 결과/오류는 빈 DF."""
    p = dict(method="getList", apiKey=KEY_KOSIS, format="json", jsonVD="Y",
             prdSe=prdSe, startPrdDe=start, endPrdDe=end,
             orgId=org, tblId=tbl, itmId=itmId, objL1=objL1)
    if objL2:
        p["objL2"] = objL2
    if objL3:
        p["objL3"] = objL3
    if objL4:
        p["objL4"] = objL4
    for k in range(retry):
        try:
            j = requests.get(KOSIS_URL, params=p, timeout=120).json()
            if isinstance(j, dict):           # error payload
                return pd.DataFrame()
            return pd.DataFrame(j)
        except Exception:
            if k == retry - 1:
                return pd.DataFrame()
            time.sleep(2)


# ----------------------------------------------------------------- FRED
def fred(series_id):
    txt = requests.get(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        timeout=60).text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["v"] != "."]
    return pd.Series(df["v"].astype(float).values, index=df["date"]).sort_index()


def mlog_diff(s):
    return 100 * np.log(s).diff()


def adf_p(x):
    from statsmodels.tsa.stattools import adfuller
    x = pd.Series(x).dropna()
    if len(x) < 16:
        return np.nan
    try:
        return round(adfuller(x, autolag="AIC")[1], 4)
    except Exception:
        return np.nan
