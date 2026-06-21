# -*- coding: utf-8 -*-
"""Phase B — 혼합주기 상태공간으로 70산업 월별 실질부가가치 잠재 시계열 추출.

절차(산업 i):
  (1) 부모(BOK-36) 분기 실질부가가치를 생산지수 비중으로 자식에 배분 → 자식 분기 VA (벤치마크)
  (2) 누적기 상태공간 + 커스텀 칼만(_mf_kalman.chowlin_kalman)으로
      분기 VA(로그)를 월별 잠재 로그수준 m_t 로 분해.
      관측: 실질주가가치(로그표준화)·생산지수(로그표준화) + 월별 외생, 상태: 월별 m_t.
  (3) latent_monthly_va(date×ind_id, 로그수준·전월비성장률) 저장 + 정합성/holdout 진단.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import read_sql, to_sql, DATA
from _mf_kalman import chowlin_kalman, zscore_log
import os

MONTHS = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
MSTR = [d.strftime("%Y-%m-01") for d in MONTHS]
T = len(MONTHS)
QUARTERS = [f"{2000+i//4}Q{i%4+1}" for i in range(104)]
QMIDX = [(2000 + i // 4 - 2000) * 12 + ((i % 4 + 1) * 3 - 1) for i in range(104)]


def _q_of_month(d):
    return f"{d.year}Q{(d.month-1)//3+1}"


def load_inputs():
    imap = read_sql("select * from clean_industry_map")
    va = read_sql("select quarter, code, value from clean_va_real")
    vapiv = va.pivot_table(index="quarter", columns="code", values="value")
    vapiv = vapiv.reindex(QUARTERS)
    mfg = read_sql("select date, ksic, value from raw_mfg_prod")
    svc = read_sql("select date, ksic, value from raw_svc_prod")
    mfgp = mfg.pivot_table(index="date", columns="ksic", values="value").reindex(MSTR)
    svcp = svc.pivot_table(index="date", columns="ksic", values="value").reindex(MSTR)
    stock = read_sql("select date, sector_code, real_stock from clean_stock_real")
    stockp = stock.pivot_table(index="date", columns="sector_code", values="real_stock").reindex(MSTR)
    exog = read_sql("select * from clean_exog_m").set_index("date").reindex(MSTR)
    return imap, vapiv, mfgp, svcp, stockp, exog


def prod_series(row, mfgp, svcp):
    """산업의 월별 생산지수(없으면 None)."""
    if row["prod_src"] == "mfg" and row["ksic"] in mfgp.columns:
        return mfgp[row["ksic"]].values.astype(float)
    if row["prod_src"] == "svc" and row["ksic"] in svcp.columns:
        return svcp[row["ksic"]].values.astype(float)
    return None


def child_quarterly_va(imap, vapiv, mfgp, svcp):
    """부모 분기 VA를 생산지수 비중으로 자식에 배분 → {ind_id: 분기 VA Series(104)}."""
    out = {}
    # 부모별 자식 그룹
    for parent_key, grp in imap.groupby("bok36"):
        codes = parent_key.split(";")
        parent_q = vapiv[codes].sum(axis=1) if all(c in vapiv for c in codes) else None
        children = grp["ind_id"].tolist()
        # 자식 분기 생산지수(월→분기평균)
        prodQ = {}
        for _, r in grp.iterrows():
            p = prod_series(r, mfgp, svcp)
            if p is None:
                prodQ[r["ind_id"]] = None
            else:
                s = pd.Series(p, index=MSTR)
                qv = s.groupby([_q_of_month(pd.Timestamp(d)) for d in MSTR]).mean()
                prodQ[r["ind_id"]] = qv.reindex(QUARTERS)
        have = [c for c in children if prodQ[c] is not None]
        # 자식별 평균비중(결측 분기 대체용): 생산지수 보유 자식은 평균지수, 미보유는 소량
        EPS = 1e-3
        meanlev = {c: (np.nanmean(prodQ[c].values) if c in have else np.nan) for c in children}
        for q_i, q in enumerate(QUARTERS):
            tot = parent_q.iloc[q_i] if parent_q is not None else np.nan
            if len(children) == 1:
                out.setdefault(children[0], pd.Series(index=QUARTERS, dtype=float))[q] = tot
                continue
            # 각 자식의 분기 가중치: 당분기 생산지수, 결측이면 평균지수, 그래도 없으면 EPS
            wts = {}
            for c in children:
                if c in have and pd.notna(prodQ[c].iloc[q_i]) and prodQ[c].iloc[q_i] > 0:
                    wts[c] = float(prodQ[c].iloc[q_i])
                elif pd.notna(meanlev[c]) and meanlev[c] > 0:
                    wts[c] = float(meanlev[c])
                else:
                    wts[c] = EPS
            denom = sum(wts.values())
            for c in children:
                sh = wts[c] / denom if denom > 0 else 1.0 / len(children)
                out.setdefault(c, pd.Series(index=QUARTERS, dtype=float))[q] = tot * sh
    return out


def build_X(row, mfgp, svcp, stockp, exog):
    """관측 회귀행렬: const, trend, 실질주가가치(로그std), 생산지수(로그std)."""
    ind = {}
    p = prod_series(row, mfgp, svcp)
    if p is not None:
        ind["prod"] = zscore_log(p)
    if row["kospi_code"] and row["kospi_code"] in stockp.columns:
        ind["stock"] = zscore_log(stockp[row["kospi_code"]].values.astype(float))
    # 외생(정상)도 관측식 보조 지표로 포함(표준화)
    ex = {}
    for c in ["oil_g", "trd_g", "d_rrate", "rfx_g", "gpr_l"]:
        if c in exog.columns:
            v = exog[c].values.astype(float)
            ex[c] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    # 결측 보간(앞뒤 채움) — 회귀행렬은 결측 불가
    cols = [np.ones(T), np.linspace(0, 1, T)]
    names = ["const", "trend"]
    for nm, v in {**ind, **ex}.items():
        s = pd.Series(v).interpolate(limit_direction="both").fillna(0).values
        cols.append(s); names.append(nm)
    return np.column_stack(cols), names


def disaggregate(row, qva, mfgp, svcp, stockp, exog, mask_last=0):
    """자식 분기 VA(로그)를 월별 로그수준으로 분해. mask_last>0이면 holdout."""
    y = qva.astype(float).copy()
    if (y.dropna() <= 0).any() or y.dropna().empty:
        return None
    ylog = np.log(y.values)
    qmidx = list(QMIDX)
    qy = ylog.copy()
    masked = None
    if mask_last > 0:
        masked = qy[-mask_last:].copy()
        qy[-mask_last:] = np.nan
    X, names = build_X(row, mfgp, svcp, stockp, exog)
    res = chowlin_kalman(qy, qmidx, X)
    res["names"] = names
    res["masked"] = masked
    return res


def main():
    imap, vapiv, mfgp, svcp, stockp, exog = load_inputs()
    cq = child_quarterly_va(imap, vapiv, mfgp, svcp)

    recs = []; diag = []; hold = []
    for _, row in imap.iterrows():
        iid = row["ind_id"]
        qva = cq.get(iid)
        if qva is None:
            print("  [skip]", iid); continue
        res = disaggregate(row, qva, mfgp, svcp, stockp, exog)
        if res is None:
            print("  [skip-neg]", iid); continue
        m = res["m"]                      # 월별 로그수준
        lvl = np.exp(m)
        g = np.r_[np.nan, 100 * np.diff(m)]   # 전월비 성장률(%)
        for t in range(T):
            recs.append((MSTR[t], iid, float(m[t]), float(lvl[t]), float(g[t]) if t else np.nan))
        # 정합성: 월→분기평균(로그) vs 입력 분기 로그
        qfit = pd.Series(m, index=MSTR).groupby([_q_of_month(pd.Timestamp(d)) for d in MSTR]).mean().reindex(QUARTERS)
        qin = np.log(qva.astype(float))
        err = (qfit.values - qin.values)
        maxerr = np.nanmax(np.abs(err))
        diag.append((iid, row["ind_name"], round(res["rho"], 3), round(res["s2"], 5),
                     round(float(maxerr), 5), int(res["m"].shape[0])))
        # holdout (생산지수 보유 산업만, 마지막 8분기)
        if row["prod_src"] in ("mfg", "svc"):
            hr = disaggregate(row, qva, mfgp, svcp, stockp, exog, mask_last=8)
            if hr is not None and hr["masked"] is not None:
                qfit2 = pd.Series(hr["m"], index=MSTR).groupby(
                    [_q_of_month(pd.Timestamp(d)) for d in MSTR]).mean().reindex(QUARTERS)
                pred = qfit2.values[-8:]; act = hr["masked"]
                rmse = float(np.sqrt(np.nanmean((pred - act) ** 2)))
                corr = float(np.corrcoef(pred[~np.isnan(act)], act[~np.isnan(act)])[0, 1]) if np.sum(~np.isnan(act)) > 2 else np.nan
                hold.append((iid, round(rmse, 4), round(corr, 3)))

    latent = pd.DataFrame(recs, columns=["date", "ind_id", "logva", "va_level", "va_growth"])
    to_sql(latent, "latent_monthly_va")
    dg = pd.DataFrame(diag, columns=["ind_id", "ind_name", "rho", "s2", "max_recon_err", "T"])
    to_sql(dg, "latent_diagnostics")
    hd = pd.DataFrame(hold, columns=["ind_id", "holdout_rmse", "holdout_corr"])
    to_sql(hd, "latent_holdout")
    latent.to_csv(os.path.join(DATA, "latent_monthly_va.csv"), index=False, encoding="utf-8-sig")
    dg.to_csv(os.path.join(DATA, "latent_diagnostics.csv"), index=False, encoding="utf-8-sig")

    print(f"latent_monthly_va: {len(latent)} rows | 산업 {latent['ind_id'].nunique()} | {latent['date'].min()}~{latent['date'].max()}")
    print(f"정합성 max_recon_err: 평균 {dg['max_recon_err'].mean():.5f}, 최대 {dg['max_recon_err'].max():.5f}")
    print(f"rho 분포: 평균 {dg['rho'].mean():.3f} [{dg['rho'].min():.2f},{dg['rho'].max():.2f}]")
    if len(hd):
        print(f"holdout(8q): RMSE 평균 {hd['holdout_rmse'].mean():.4f} | corr 평균 {hd['holdout_corr'].mean():.3f} (n={len(hd)})")
    print("DONE")


if __name__ == "__main__":
    main()
