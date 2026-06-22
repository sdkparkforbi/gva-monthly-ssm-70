# -*- coding: utf-8 -*-
"""Phase D — S1~S5 시나리오 전망(월별) + 몬테카를로 팬차트.

선행연구와 동일 설계(노동공급 중위/저위/고위 × 대외환경 2축), 월별로 운용.
  S1 기준 / S2 제약+위기 / S3 제약완화 / S4 기술도약 / S5 지정학위기
전망지평: 2026-01 ~ 2035-12 (120개월). VARX 점추정 + 잔차 몬테카를로(400 draw).
"""
import sys, json
import numpy as np
import pandas as pd
import importlib
sys.path.insert(0, ".")
from _common import read_sql, to_sql, DATA
import os

v5 = importlib.import_module("05_varx_estimate")
EXOG = v5.EXOG
P, Q = v5.P, v5.Q
HMON = 120
FUT = pd.date_range("2026-01-01", periods=HMON, freq="MS")
NDRAW = 400
np.random.seed(7)

SCN = ["S1", "S2", "S3", "S4", "S5"]


def exog_paths():
    """거시 블록(DSGE, 08) 분기 외생경로를 월별 VARX 외생으로 연결.
    DSGE 분기 성장/변화율 → 월별 = 분기값/3 (3개월 합 = 분기값). GPR은 역사적 기준값."""
    base = read_sql("select * from clean_exog_m")[EXOG].mean().to_dict()
    dz = read_sql("select * from result_dsge_exog")        # scenario, quarter, oil_g..lab_g
    paths = {}
    for scn in SCN:
        d = dz[dz.scenario == scn].reset_index(drop=True)  # 40분기
        X = np.zeros((HMON, len(EXOG)))
        for t in range(HMON):
            q = min(t // 3, len(d) - 1)
            for j, v in enumerate(EXOG):
                if v in d.columns:
                    X[t, j] = float(d.loc[q, v]) / 3.0     # 분기→월
                else:                                       # gpr_l: DSGE 미산출 → 기준값
                    X[t, j] = base.get(v, 0.0)
        paths[scn] = X
    return paths, list(EXOG)


def weights_2024():
    """70 MECE leaf 가중치 = 2024 평균 실질 부가가치 수준(leaf 합=부모, MECE 정합)."""
    lv = read_sql("select date, ind_id, va_level from latent_monthly_va")
    w = lv[lv["date"].str.startswith("2024")].groupby("ind_id")["va_level"].mean()
    return (w / w.sum())


def main():
    # VARX 재추정(점추정 + 잔차공분산)
    Y, ex, inds = v5.load()
    X, Yt, n, ne = v5.design(Y, ex)
    lam = v5.select_lambda(X, Yt, n)
    B = v5.ridge_fit(X, Yt, lam, n)
    resid = Yt - X @ B
    Sigma = np.cov(resid.T)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))

    paths, exn = exog_paths()
    w = weights_2024().reindex(inds).fillna(0).values

    yhist = Y.values[-P:]                 # 시드 내생(마지막 P개월 성장률)
    xhist = ex.values[-Q:]               # 시드 외생

    # YoY 기준: 실제 2025년 월별 총계지수(2025.12=100)
    lv25 = read_sql("select date, ind_id, va_level from latent_monthly_va where date like '2025-%'")
    piv25 = lv25.pivot(index="date", columns="ind_id", values="va_level").sort_index().reindex(columns=inds)
    hist25 = ((piv25 / piv25.loc["2025-12-01"] * 100).values * w).sum(axis=1)   # 12개월
    YEARS = list(range(2026, 2036))

    def annual_yoy(path120):
        full = np.concatenate([hist25, path120])          # 132개월: 2025.01~2035.12
        ann = full.reshape(11, 12).mean(axis=1)           # 연평균 2025~2035
        return (ann[1:] / ann[:-1] - 1) * 100             # YoY 2026~2035

    def simulate(scn, draws=0):
        Xf = paths[scn]
        ylag = list(yhist[::-1])         # ylag[0]=t-1
        xfull = np.vstack([xhist, Xf])   # 과거 Q + 미래
        levels = np.zeros((HMON, n))
        cum = np.zeros(n)
        for t in range(HMON):
            xt = [xfull[Q + t - s] for s in range(0, Q + 1)]
            reg = np.concatenate([[1.0]] + ylag[:P] + xt)
            yt = reg @ B
            if draws:
                yt = yt + (L @ np.random.randn(n))
            ylag = [yt] + ylag
            cum += yt                     # 누적 성장률(로그%)
            levels[t] = 100 * np.exp(cum / 100.0)
        return levels

    fan_rows = []; ind_rows = []; grow_rows = []
    aggfan = {}
    for scn in SCN:
        pt = simulate(scn, draws=0)
        agg_pt = (pt * w).sum(axis=1)
        # 산업별 2035 누적(기준 마지막관측월=100)
        for i, iid in enumerate(inds):
            ind_rows.append((scn, iid, round(float(pt[-1, i] - 100), 2)))
        # 팬차트(총계): 400 draw
        sims = np.array([(simulate(scn, draws=1) * w).sum(axis=1) for _ in range(NDRAW)])
        pct = np.percentile(sims, [5, 15, 50, 85, 95], axis=0)
        for t, d in enumerate(FUT):
            fan_rows.append((scn, d.strftime("%Y-%m-01"),
                             round(agg_pt[t], 2), round(pct[0, t], 2), round(pct[1, t], 2),
                             round(pct[2, t], 2), round(pct[3, t], 2), round(pct[4, t], 2)))
        aggfan[scn] = dict(median=agg_pt.tolist())
        # 전년대비 증가율(YoY) 팬: draw별 YoY → 분위수
        g_pt = annual_yoy(agg_pt)
        gsims = np.array([annual_yoy(s) for s in sims])
        gpct = np.percentile(gsims, [5, 50, 95], axis=0)
        for k, yr in enumerate(YEARS):
            grow_rows.append((scn, yr, round(float(g_pt[k]), 2),
                              round(float(gpct[0, k]), 2), round(float(gpct[1, k]), 2),
                              round(float(gpct[2, k]), 2)))

    fan = pd.DataFrame(fan_rows, columns=["scenario", "date", "point", "p5", "p15", "p50", "p85", "p95"])
    ind = pd.DataFrame(ind_rows, columns=["scenario", "ind_id", "cum_growth_2035_pct"])
    grow = pd.DataFrame(grow_rows, columns=["scenario", "year", "g_point", "g_p5", "g_p50", "g_p95"])
    to_sql(fan, "result_scenario_fan"); to_sql(ind, "result_scenario_industry_2035")
    to_sql(grow, "result_scenario_growth")
    grow.to_csv(os.path.join(DATA, "scenario_gdp_growth.csv"), index=False, encoding="utf-8-sig")
    fan.to_csv(os.path.join(DATA, "scenario_gdp_fan.csv"), index=False, encoding="utf-8-sig")
    ind.to_csv(os.path.join(DATA, "scenario_industry_2035.csv"), index=False, encoding="utf-8-sig")

    # 노동효과(S3-S2) 산업별
    piv = ind.pivot(index="ind_id", columns="scenario", values="cum_growth_2035_pct")
    piv["labor_effect_S3_S2"] = piv["S3"] - piv["S2"]
    piv.reset_index().to_csv(os.path.join(DATA, "scenario_labor_effect.csv"),
                             index=False, encoding="utf-8-sig")

    # 연말 총계지수(연도별)
    yearly = fan.copy()
    yearly["year"] = yearly["date"].str[:4]
    yend = yearly.groupby(["scenario", "year"]).last().reset_index()
    summary = {scn: {"2035_point": float(fan[(fan.scenario == scn)].iloc[-1]["point"]),
                     "2035_p5": float(fan[(fan.scenario == scn)].iloc[-1]["p5"]),
                     "2035_p95": float(fan[(fan.scenario == scn)].iloc[-1]["p95"])}
               for scn in SCN}
    print("시나리오 2035 총부가가치 지수(2025.12=100):")
    for scn in SCN:
        s = summary[scn]
        print(f"  {scn}: {s['2035_point']:.1f}  [{s['2035_p5']:.1f}, {s['2035_p95']:.1f}]")
    top = piv.sort_values("labor_effect_S3_S2", ascending=False)
    print("노동효과(S3-S2) 민감 상위:", top.head(3).index.tolist())
    json.dump({"fan": aggfan, "summary": summary,
               "industry": ind.pivot(index="ind_id", columns="scenario",
                                      values="cum_growth_2035_pct").reset_index().to_dict("records")},
              open(os.path.join(DATA, "_scenario_viz.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("DONE")


if __name__ == "__main__":
    main()
