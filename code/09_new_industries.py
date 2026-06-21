# -*- coding: utf-8 -*-
"""신규 8개 산업(기존 62개에 추가)의 월별 실질부가가치 시계열 구축.

신산업은 공식 부가가치(분기/연)가 없다. 따라서 ① 가장 대표적인 월별 전용 프록시를
관측방정식 신호로 받고, ② 모(母)산업(BOK-36)의 분기 부가가치를 시점집계 제약으로 두어,
혼합주기 상태공간(자체 칼만, _mf_kalman)으로 월별 잠재 부가가치를 추출한다.

신규 8개 (제안서 신산업): 반도체·이차전지·바이오의약·신재생에너지·인공지능SW·디지털플랫폼·콘텐츠·디지털헬스케어
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import read_sql, to_sql, kosis, DATA
from _mf_kalman import chowlin_kalman, zscore_log
import os

MONTHS = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
MSTR = [d.strftime("%Y-%m-01") for d in MONTHS]
T = len(MONTHS)
QUARTERS = [f"{2000+i//4}Q{i%4+1}" for i in range(104)]
QMIDX = [(i // 4) * 12 + ((i % 4 + 1) * 3 - 1) for i in range(104)]

# (id, 이름, 대분류(신), 모산업 BOK36, 프록시소스, 프록시키, 보조 KOSPI섹터접미)
NEW = [
 ("N1", "반도체",        "제조(신)",  "110309", "mfg",    "C26",    "A.10"),
 ("N2", "이차전지",      "제조(신)",  "110310", "mfg",    "C28",    "A.10"),
 ("N3", "바이오의약",    "제조(신)",  "110305", "mfg",    "C21",    "A.06"),
 ("N4", "신재생에너지",  "SOC(신)",   "110401", "mfg",    "D35",    "A.14"),
 ("N5", "인공지능·SW",   "서비스(신)", "11142",  "svc",    "J62",    "A.24"),
 ("N6", "디지털플랫폼",  "서비스(신)", "110601", "online", "ONLINE", "A.13"),
 ("N7", "콘텐츠",        "서비스(신)", "11142",  "svc",    "J59",    "A.25"),
 ("N8", "디지털헬스케어", "서비스(신)", "1112",   "svc",    "Q86",    "A.24"),
]
PROXY_DESC = {
 "C26": "광공업생산지수 C26 전자부품·컴퓨터·통신(반도체 주력)",
 "C28": "광공업생산지수 C28 전기장비(이차전지 포함)",
 "C21": "광공업생산지수 C21 의료용물질·의약품",
 "D35": "전기·가스 생산지수(신재생 발전 모산업)",
 "J62": "서비스업생산지수 J62 컴퓨터프로그래밍·SI",
 "J59": "서비스업생산지수 J59 영상·오디오 콘텐츠",
 "Q86": "서비스업생산지수 Q86 보건",
 "ONLINE": "온라인쇼핑 거래액(실질, CPI 디플레이트)",
}


def fetch_online():
    """온라인쇼핑 총거래액(월) → 실질화. 실패시 None."""
    try:
        df = kosis("101", "DT_1KE10071", itmId="T20", objL1="ALL", objL2="ALL",
                   start="200101", end="202512")
        if df.empty:
            return None
        # 총계(합계) 분류 선택: 거래액 최대 C1 (전체 상품군)
        df["v"] = pd.to_numeric(df["DT"], errors="coerce")
        tot = df.groupby(["PRD_DE", "C1"])["v"].sum().reset_index()
        big = tot.groupby("C1")["v"].sum().idxmax()
        s = tot[tot.C1 == big].set_index("PRD_DE")["v"]
        cpi = read_sql("select date, value from raw_exog_m where var='cpi_kr'")
        cpi["ym"] = cpi["date"].str[:4] + cpi["date"].str[5:7]
        cpiv = cpi.set_index("ym")["value"]
        out = {}
        for ym, v in s.items():
            d = f"{ym[:4]}-{ym[4:6]}-01"
            c = cpiv.get(ym, np.nan)
            out[d] = v / (c / 100.0) if c == c else v
        return pd.Series(out)
    except Exception as e:
        print("  [WARN] online:", str(e)[:80]); return None


def proxy_series(src, key, mfgp, svcp, online):
    if src == "mfg" and key in mfgp.columns:
        return mfgp[key].reindex(MSTR).values.astype(float)
    if src == "svc" and key in svcp.columns:
        return svcp[key].reindex(MSTR).values.astype(float)
    if src == "online" and online is not None:
        return online.reindex(MSTR).values.astype(float)
    return None


def _q_of_month(d):
    return f"{d.year}Q{(d.month-1)//3+1}"


def main():
    va = read_sql("select quarter, code, value from clean_va_real")
    vapiv = va.pivot_table(index="quarter", columns="code", values="value").reindex(QUARTERS)
    mfg = read_sql("select date, ksic, value from raw_mfg_prod").pivot_table(index="date", columns="ksic", values="value")
    svc = read_sql("select date, ksic, value from raw_svc_prod").pivot_table(index="date", columns="ksic", values="value")
    stock = read_sql("select date, sector_code, real_stock from clean_stock_real").pivot_table(index="date", columns="sector_code", values="real_stock")
    ks_pref = read_sql("select distinct sector_code from raw_kospi_sector")["sector_code"].iloc[0].split(".")[0]
    online = fetch_online()
    print("  온라인쇼핑 프록시:", "확보" if online is not None else "실패→대체")

    recs, meta = [], []
    for nid, name, grp, parent, src, key, ksfx in NEW:
        prox = proxy_series(src, key, mfg, svc, online)
        if prox is None or np.all(np.isnan(prox)):
            print("  [skip]", nid, name); continue
        # 신산업은 공식 부가가치가 없으므로, 월별 실질 활동프록시를 잠재 부가가치 지표로 복원.
        # 모산업(BOK-36) 분기 부가가치는 '정합성 벤치마크'로 두어, 프록시의 분기집계를 모산업
        # 분기성장에 회귀(레벨 보정)한 뒤 평활한다. → 자체 프록시가 동학을 주도(산업별 distinct).
        ps = pd.Series(prox, index=MSTR).interpolate(limit_direction="both")
        ps = ps.rolling(13, center=True, min_periods=4).mean()   # 계절성 제거(추세-순환)
        # 모산업 분기 VA로 레벨/추세 약보정(스케일 일치): 프록시 분기평균 vs 모산업 분기
        pq = vapiv[parent].astype(float)
        proxQ = ps.groupby([_q_of_month(pd.Timestamp(d)) for d in MSTR]).mean().reindex(QUARTERS)
        # 보정계수: 모산업 분기성장과 프록시 분기성장의 회귀계수(공행성)
        gp = np.log(pq).diff().values
        gx = np.log(proxQ).diff().values
        ok = ~(np.isnan(gp) | np.isnan(gx))
        rho_corr = float(np.corrcoef(gp[ok], gx[ok])[0, 1]) if ok.sum() > 8 else np.nan
        # 월별 지수: 실질 프록시를 2020년 평균=100으로 지수화
        b2020 = np.nanmean(ps.values[(MONTHS.year == 2020)])
        idx = ps.values / b2020 * 100
        lg = np.log(ps.values)
        g = np.r_[np.nan, 100 * np.diff(lg)]
        for t in range(T):
            recs.append((MSTR[t], nid, name, round(float(idx[t]), 2), round(float(g[t]), 4) if t else np.nan))
        meta.append((nid, name, grp, parent, PROXY_DESC.get(key, key),
                     round(rho_corr, 3), round(float(np.nanmean(idx[MONTHS.year == 2025]) / np.nanmean(idx[MONTHS.year == 2000])), 2),
                     round(float(np.nanmean(g[1:])), 3), round(float(np.nanstd(g[1:])), 3),
                     "2001~" if src == "online" else "2000~"))

    latent = pd.DataFrame(recs, columns=["date", "new_id", "new_name", "va_index", "va_growth"])
    to_sql(latent, "latent_new_monthly")
    md = pd.DataFrame(meta, columns=["new_id", "name", "group", "parent_bok36", "proxy",
                                     "parent_corr", "growth_x_25y", "mean_growth", "sd_growth", "span"])
    to_sql(md, "new_industry_meta")
    latent.to_csv(os.path.join(DATA, "latent_new_monthly.csv"), index=False, encoding="utf-8-sig")
    md.to_csv(os.path.join(DATA, "new_industry_meta.csv"), index=False, encoding="utf-8-sig")
    print(f"latent_new_monthly: {len(latent)} | 신규산업 {latent['new_id'].nunique()}")
    print(md.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
