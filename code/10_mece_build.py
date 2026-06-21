# -*- coding: utf-8 -*-
"""MECE 재구조화: 62 base + 8 carve = 70 (상호배타·전체포괄).

원칙: 각 BOK-36 부모 안에서 leaf 부가가치 비중의 합 = 1 (→ 합산 시 경제 전체 보존).
  · 62 base = 표준 산업(일부 KSIC 병합) 중 8개는 'host'.
  · 8 emerging = host에서 기준연도(2020) 공개통계 비중으로 분할. host는 잔여(residual)로 축소.
  · host_m = emerging_m + residual_m  (가법성 → MECE 보존).
신산업 전용 프록시가 host와 다르면 시변(時變) 비중으로 자체 동학 부여, 같으면 상수 비중.

산출(덮어쓰기): clean_industry_map(70, role/host/share 포함), latent_monthly_va(70 MECE),
              latent_diagnostics, + data CSV. MECE 재구성오차 검증.
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


def qof(d): return f"{d.year}Q{(d.month-1)//3+1}"


# ── 62 structural units (id, name, sector, bok36(';'), prod_src, prod_keys(list), kospi_sfx, is_host)
STRUCT = [
 ("AGR", "농림어업", "농림어업", "1101", "none", [], "", False),
 ("MIN_COAL", "석탄·원유·천연가스 광업", "광업", "1102", "mfg", ["B05"], "", False),
 ("MIN_MET", "금속 광업", "광업", "1102", "mfg", ["B06"], "", False),
 ("MIN_NONMET", "비금속광물 광업", "광업", "1102", "mfg", ["B07"], "", False),
 ("MFG_FOOD", "식료품", "제조업", "110301", "mfg", ["C10"], "A.02", False),
 ("MFG_BEV", "음료", "제조업", "110301", "mfg", ["C11"], "A.02", False),
 ("MFG_TOB", "담배", "제조업", "110301", "mfg", ["C12"], "A.02", False),
 ("MFG_TEX", "섬유제품", "제조업", "110302", "mfg", ["C13"], "A.03", False),
 ("MFG_APP", "의복·액세서리·모피", "제조업", "110302", "mfg", ["C14"], "A.03", False),
 ("MFG_LEATHER", "가죽·가방·신발", "제조업", "110302", "mfg", ["C15"], "A.03", False),
 ("MFG_WOOD", "목재·나무제품", "제조업", "110303", "mfg", ["C16"], "A.04", False),
 ("MFG_PAPER", "펄프·종이", "제조업", "110303", "mfg", ["C17"], "A.04", False),
 ("MFG_PRINT", "인쇄·기록매체", "제조업", "110303", "mfg", ["C18"], "A.04", False),
 ("MFG_COKE", "코크스·석유정제", "제조업", "110304", "mfg", ["C19"], "A.05", False),
 ("H_CHEM", "화학·의약·고무플라스틱", "제조업", "110305", "mfg", ["C20", "C21", "C22"], "A.05", True),   # host→바이오
 ("MFG_NONMET", "비금속 광물제품", "제조업", "110306", "mfg", ["C23"], "A.07", False),
 ("MFG_STEEL", "1차 금속", "제조업", "110307", "mfg", ["C24"], "A.08", False),
 ("MFG_METAL", "금속가공제품", "제조업", "110308", "mfg", ["C25"], "A.08", False),
 ("H_ELEC", "전자부품·컴퓨터·통신장비", "제조업", "110309", "mfg", ["C26"], "A.10", True),               # host→반도체
 ("MFG_PREC", "의료·정밀·광학기기", "제조업", "110309", "mfg", ["C27"], "A.11", False),
 ("H_ELECEQ", "전기장비", "제조업", "110310", "mfg", ["C28"], "A.10", True),                              # host→이차전지
 ("MFG_MACH", "기타 기계·장비", "제조업", "110311", "mfg", ["C29"], "A.09", False),
 ("MFG_AUTO", "자동차·트레일러", "제조업", "110312", "mfg", ["C30"], "A.12", False),
 ("MFG_OTRANS", "기타 운송장비", "제조업", "110312", "mfg", ["C31"], "A.12", False),
 ("MFG_FURN", "가구", "제조업", "110313", "mfg", ["C32"], "", False),
 ("MFG_OTH", "기타 제품", "제조업", "110313", "mfg", ["C33"], "", False),
 ("MFG_REP", "산업용 기계 수리", "제조업", "110313", "mfg", ["C34"], "", False),
 ("H_UTIL", "전기·가스·증기", "SOC", "110401;110402", "mfg", ["D35"], "A.14", True),                      # host→신재생
 ("UTL_WATER", "수도", "SOC", "110403", "svc", ["E36"], "", False),
 ("UTL_SEW", "하수·폐수처리", "SOC", "110403", "svc", ["E37"], "", False),
 ("UTL_WASTE", "폐기물·원료재생", "SOC", "110403", "svc", ["E38"], "", False),
 ("CON", "건설업", "SOC", "11051;11052;11053;11054", "none", [], "A.15", False),
 ("H_RETAIL", "도소매", "서비스업", "110601", "svc", ["G45", "G46", "G47"], "A.13", True),                # host→디지털플랫폼
 ("TRN_LAND", "육상·파이프라인 운송", "서비스업", "1107", "svc", ["H49"], "A.16", False),
 ("TRN_SEA", "수상 운송", "서비스업", "1107", "svc", ["H50"], "A.16", False),
 ("TRN_AIR", "항공 운송", "서비스업", "1107", "svc", ["H51"], "A.16", False),
 ("TRN_WARE", "창고·운송서비스", "서비스업", "1107", "svc", ["H52"], "A.16", False),
 ("ACF_LODGE", "숙박", "서비스업", "110602", "svc", ["I55"], "A.22", False),
 ("ACF_FOOD", "음식점·주점", "서비스업", "110602", "svc", ["I56"], "A.22", False),
 ("H_MEDIA", "출판·영상·방송", "서비스업", "11142", "svc", ["J58", "J59", "J60"], "A.25", True),          # host→콘텐츠
 ("ICT_TEL", "우편·통신", "서비스업", "11141", "svc", ["J61"], "A.17", False),
 ("H_INFOSVC", "정보서비스(SW·IT)", "서비스업", "11142", "svc", ["J62", "J63"], "A.24", True),            # host→AI
 ("FIN_BANK", "금융업", "서비스업", "1108", "svc", ["K64"], "A.19", False),
 ("FIN_INS", "보험·연금", "서비스업", "1108", "svc", ["K65"], "A.21", False),
 ("FIN_AUX", "금융보험서비스", "서비스업", "1108", "svc", ["K66"], "A.20", False),
 ("RE", "부동산업", "서비스업", "1109", "svc", ["L68"], "A.23", False),
 ("PRO_RND", "연구개발", "서비스업", "111501", "svc", ["M70"], "A.22", False),
 ("PRO_PRO", "전문서비스", "서비스업", "111501", "svc", ["M71"], "A.22", False),
 ("PRO_ENG", "건축·엔지니어링", "서비스업", "111501", "svc", ["M72"], "A.22", False),
 ("PRO_ETC", "기타 전문·과학", "서비스업", "111501", "svc", ["M73"], "A.22", False),
 ("BUS_FAC", "사업시설관리", "서비스업", "111502", "svc", ["N74"], "A.22", False),
 ("BUS_SUP", "사업지원", "서비스업", "111502", "svc", ["N75"], "A.22", False),
 ("BUS_RENT", "임대업", "서비스업", "111502", "svc", ["N76"], "A.22", False),
 ("PUB", "공공행정·국방", "서비스업", "1110", "none", [], "", False),
 ("EDU", "교육 서비스", "서비스업", "1111", "svc", ["P85"], "A.22", False),
 ("H_HEALTH", "보건업", "서비스업", "1112", "svc", ["Q86"], "A.22", True),                                # host→디지털헬스케어
 ("HW_WELF", "사회복지", "서비스업", "1112", "svc", ["Q87"], "", False),
 ("ART_CRE", "창작·예술·여가", "서비스업", "11131", "svc", ["R90"], "A.25", False),
 ("ART_SPO", "스포츠·오락", "서비스업", "11131", "svc", ["R91"], "A.25", False),
 ("PER_ASSOC", "협회·단체", "서비스업", "11132", "svc", ["S94"], "", False),
 ("PER_REP", "개인용품 수리", "서비스업", "11132", "svc", ["S95"], "", False),
 ("PER_OTH", "기타 개인서비스", "서비스업", "11132", "svc", ["S96"], "", False),
]

# ── 8 emerging (id, name, host_id, base_share(2020), em_src, em_keys, share_proxy_differs)
EMERG = [
 ("E_SEMI",  "반도체",        "H_ELEC",    0.50, "mfg",    ["C26"],    False),
 ("E_BATT",  "이차전지",      "H_ELECEQ",  0.22, "mfg",    ["C28"],    False),
 ("E_BIO",   "바이오의약",    "H_CHEM",    0.18, "mfg",    ["C21"],    True),
 ("E_RENEW", "신재생에너지",  "H_UTIL",    0.08, "mfg",    ["D35"],    False),
 ("E_AI",    "인공지능·SW",   "H_INFOSVC", 0.45, "svc",    ["J62"],    True),
 ("E_PLAT",  "디지털플랫폼",  "H_RETAIL",  0.12, "online", ["ONLINE"], True),
 ("E_CONT",  "콘텐츠",        "H_MEDIA",   0.45, "svc",    ["J59"],    True),
 ("E_HEALTH","디지털헬스케어","H_HEALTH",  0.05, "svc",    ["Q86"],    False),
]
# 기준연도 비중 출처(문서화): 광업제조업조사 부가가치(반도체·전지·의약), 온라인쇼핑동향 소매판매 비중,
# 신재생에너지 발전비중(2020), 정보통신산업통계(SW), 보건산업통계(디지털헬스).
SHARE_SRC = "2020 공개통계(광업제조업조사 부가가치·온라인쇼핑동향·신재생발전비중·정보통신/보건산업통계) 기반"


def avg_index(keys, src, mfgp, svcp, online):
    """여러 KSIC 생산지수를 평균(2020=100 정규화 후)한 프록시 시계열."""
    arrs = []
    for k in keys:
        if src == "mfg" and k in mfgp.columns:
            arrs.append(mfgp[k].reindex(MSTR).values.astype(float))
        elif src == "svc" and k in svcp.columns:
            arrs.append(svcp[k].reindex(MSTR).values.astype(float))
        elif src == "online" and online is not None:
            arrs.append(online.reindex(MSTR).values.astype(float))
    if not arrs:
        return None
    A = []
    for a in arrs:
        b = np.nanmean(a[MONTHS.year == 2020])
        A.append(a / b * 100 if b and b == b else a)
    return np.nanmean(np.vstack(A), axis=0)


def fetch_online():
    try:
        df = kosis("101", "DT_1KE10071", itmId="T20", objL1="ALL", objL2="ALL", start="200101", end="202512")
        if df.empty: return None
        df["v"] = pd.to_numeric(df["DT"], errors="coerce")
        tot = df.groupby(["PRD_DE", "C1"])["v"].sum().reset_index()
        big = tot.groupby("C1")["v"].sum().idxmax()
        s = tot[tot.C1 == big].set_index("PRD_DE")["v"]
        cpi = read_sql("select date, value from raw_exog_m where var='cpi_kr'")
        cpi["ym"] = cpi["date"].str[:4] + cpi["date"].str[5:7]
        cpiv = cpi.set_index("ym")["value"]
        out = {f"{ym[:4]}-{ym[4:6]}-01": (v / (cpiv.get(ym, np.nan) / 100.0)) for ym, v in s.items()}
        return pd.Series(out)
    except Exception:
        return None


def disagg(qva_log, X):
    res = chowlin_kalman(qva_log, list(QMIDX), X)
    return res


def main():
    va = read_sql("select quarter, code, value from clean_va_real")
    vapiv = va.pivot_table(index="quarter", columns="code", values="value").reindex(QUARTERS)
    mfgp = read_sql("select date,ksic,value from raw_mfg_prod").pivot_table(index="date", columns="ksic", values="value")
    svcp = read_sql("select date,ksic,value from raw_svc_prod").pivot_table(index="date", columns="ksic", values="value")
    stock = read_sql("select date,sector_code,real_stock from clean_stock_real").pivot_table(index="date", columns="sector_code", values="real_stock")
    kspref = read_sql("select distinct sector_code from raw_kospi_sector")["sector_code"].iloc[0].split(".")[0]
    online = fetch_online()
    print("  online:", "ok" if online is not None else "fallback")

    # structural proxy + parent grouping
    struct = {s[0]: dict(name=s[1], sector=s[2], bok=s[3], src=s[4], keys=s[5], ks=s[6], host=s[7]) for s in STRUCT}
    for sid, s in struct.items():
        s["prox"] = avg_index(s["keys"], s["src"], mfgp, svcp, online) if s["keys"] else None
    parents = {}
    for sid, s in struct.items():
        parents.setdefault(s["bok"], []).append(sid)

    # 1) 부모 분기 VA → structural units 비중배분 (2020 프록시 평균 수준, 없으면 균등)
    structQ = {}
    for bok, units in parents.items():
        codes = bok.split(";")
        pq = vapiv[codes].sum(axis=1)
        w = {}
        for u in units:
            p = struct[u]["prox"]
            w[u] = np.nanmean(p[MONTHS.year == 2020]) if p is not None else np.nan
        if all(np.isnan(list(w.values()))):
            for u in units: w[u] = 1.0
        mean_known = np.nanmean([v for v in w.values() if v == v]) or 1.0
        for u in units:
            if not (w[u] == w[u]): w[u] = mean_known
        tot = sum(w.values())
        for u in units:
            structQ[u] = pq * (w[u] / tot)

    # 2) structural units 월별 분해
    structM = {}
    for sid, s in struct.items():
        q = structQ[sid].astype(float)
        if (q.dropna() <= 0).any() or q.dropna().empty:
            q = q.clip(lower=max(q.max() * 1e-4, 1e-6))
        ylog = np.log(q.values)
        cols = [np.ones(T)]; names = ["const"]
        if s["prox"] is not None:
            cols.append(zscore_log(s["prox"])); names.append("prox")
        if s["ks"] and f"{kspref}.{s['ks'].split('.')[1]}" in stock.columns:
            cols.append(zscore_log(stock[f"{kspref}.{s['ks'].split('.')[1]}"].reindex(MSTR).values.astype(float))); names.append("stock")
        if len(cols) == 1:
            cols.append(np.linspace(0, 1, T)); names.append("trend")
        X = np.column_stack([pd.Series(c).interpolate(limit_direction="both").fillna(0).values for c in cols])
        res = disagg(ylog, X)
        structM[sid] = dict(m=res["m"], lvl=np.exp(res["m"]), rho=res["rho"])

    # 3) host 분할 → emerging + residual (시변/상수 비중)
    leaves = {}   # leaf_id -> dict(name, sector, bok, role, host, lvl(monthly), rho)
    host_share = {}
    emer = {e[0]: dict(name=e[1], host=e[2], share=e[3], src=e[4], keys=e[5], tv=e[6]) for e in EMERG}
    em_by_host = {e["host"]: eid for eid, e in emer.items()}

    for sid, s in struct.items():
        lvl = structM[sid]["lvl"]
        if sid in em_by_host:
            eid = em_by_host[sid]; e = emer[eid]
            base = e["share"]
            if e["tv"]:
                ep = avg_index(e["keys"], e["src"], mfgp, svcp, online)
                hp = s["prox"]
                ep = pd.Series(ep, index=MSTR).interpolate(limit_direction="both").values
                hp = pd.Series(hp, index=MSTR).interpolate(limit_direction="both").values
                e20 = np.nanmean(ep[MONTHS.year == 2020]); h20 = np.nanmean(hp[MONTHS.year == 2020])
                r = (ep / e20) / (hp / h20)
                s_t = np.clip(base * r, 0.01, 0.95)
            else:
                s_t = np.full(T, base)
            em_lvl = lvl * s_t
            res_lvl = lvl * (1 - s_t)
            leaves[eid] = dict(name=e["name"], sector="신산업", bok=s["bok"], role="emerging",
                               host=sid, lvl=em_lvl, rho=structM[sid]["rho"])
            leaves[sid] = dict(name=s["name"] + "(잔여)", sector=s["sector"], bok=s["bok"], role="host_residual",
                               host=sid, lvl=res_lvl, rho=structM[sid]["rho"])
            host_share[eid] = float(np.nanmean(s_t[MONTHS.year == 2020]))
        elif not s["host"]:
            leaves[sid] = dict(name=s["name"], sector=s["sector"], bok=s["bok"], role="base",
                               host="", lvl=lvl, rho=structM[sid]["rho"])

    # 4) 저장: clean_industry_map(70) + latent_monthly_va(70)
    recs, maprows, diag = [], [], []
    for lid, lf in leaves.items():
        m = np.log(np.clip(lf["lvl"], 1e-9, None))
        g = np.r_[np.nan, 100 * np.diff(m)]
        for t in range(T):
            recs.append((MSTR[t], lid, float(lf["lvl"][t]), float(g[t]) if t else np.nan))
        maprows.append((lid, lf["name"], lf["sector"], lf["bok"], lf["role"], lf["host"]))
        diag.append((lid, lf["name"], round(float(lf["rho"]), 3),
                     round(float(np.nanmean(g[1:])), 3), round(float(np.nanstd(g[1:])), 3)))
    latent = pd.DataFrame(recs, columns=["date", "ind_id", "va_level", "va_growth"])
    to_sql(latent, "latent_monthly_va")
    imap = pd.DataFrame(maprows, columns=["ind_id", "ind_name", "sector", "bok36", "role", "host"])
    to_sql(imap, "clean_industry_map")
    dg = pd.DataFrame(diag, columns=["ind_id", "ind_name", "rho", "mean_growth", "sd_growth"])
    to_sql(dg, "latent_diagnostics")
    em = pd.DataFrame([(eid, emer[eid]["name"], emer[eid]["host"], struct[emer[eid]["host"]]["name"],
                       round(host_share.get(eid, emer[eid]["share"]), 3), emer[eid]["share"]) for eid in emer],
                      columns=["em_id", "name", "host_id", "host_name", "share_2020_eff", "share_2020_assumed"])
    to_sql(em, "emerging_meta")
    imap.to_csv(os.path.join(DATA, "industry_map_70_mece.csv"), index=False, encoding="utf-8-sig")
    em.to_csv(os.path.join(DATA, "emerging_meta.csv"), index=False, encoding="utf-8-sig")

    # 5) MECE 검증: leaf 월별 합 (부모별) == 부모 월별 재구성 (structural 합)
    leafM = pd.DataFrame({lid: lf["lvl"] for lid, lf in leaves.items()}, index=MSTR)
    structSum = pd.DataFrame({sid: structM[sid]["lvl"] for sid in struct if (not struct[sid]["host"]) or sid in em_by_host}, index=MSTR)
    # 부모별 비교
    errs = []
    bok_of = {lid: lf["bok"] for lid, lf in leaves.items()}
    for bok in parents:
        lids = [lid for lid in leaves if leaves[lid]["bok"] == bok]
        sids = parents[bok]
        ls = leafM[lids].sum(axis=1); ss = structSum[[s for s in sids]].sum(axis=1)
        errs.append(float(np.nanmax(np.abs(ls - ss) / (ss.abs() + 1e-9))))
    print(f"clean_industry_map: {len(imap)} leaves | role {imap['role'].value_counts().to_dict()}")
    print(f"latent_monthly_va: {len(latent)} | 산업 {latent['ind_id'].nunique()}")
    print(f"MECE 검증(부모별 leaf합 vs structural합) 최대 상대오차: {max(errs):.2e}")
    print(em.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
