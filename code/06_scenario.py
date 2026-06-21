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

# 시나리오 노동(WAP) 추계 사용 라벨
LAB = {"S1": "중위", "S2": "저위", "S3": "고위", "S4": "중위", "S5": "중위"}
# 대외 충격(연 %p 가산, 36개월 선형 감쇠 → 일시적). 신뢰 채널 trd_g(+)·d_rrate(−)·rfx_g 중심.
SHOCK = {
 "S1": {},
 "S2": {"trd_g": -4.0, "d_rrate": +0.8, "rfx_g": +6.0},        # 제약+대외위기
 "S3": {},                                                     # 제약완화(노동 고위)
 "S4": {"trd_g": +3.0, "d_rrate": -0.3},                       # 기술도약(교역 확대)
 "S5": {"trd_g": -3.5, "d_rrate": +1.0, "gpr_l": +25.0},       # 지정학 위기
}


def lab_annual(scn):
    """시나리오별 연 취업자증가율(%) 경로 (WAP성장 + 참여율추세)."""
    w = pd.read_csv(os.path.join(DATA, "wap_projection.csv"))
    col = {"중위": "중위_천명", "저위": "저위_천명", "고위": "고위_천명"}[LAB[scn]]
    s = w.set_index("연도")[col].dropna()
    g = (s / s.shift(1) - 1) * 100
    out = {}
    for yi, y in enumerate(range(2026, 2036)):
        wapg = g.get(y, g.dropna().iloc[-1])         # 추계 종료후 마지막값 유지
        partic = max(0.9 - 0.09 * yi, 0.0)           # 참여율 추세(점감)
        out[y] = float(wapg + partic)
    return out


def exog_paths():
    base = read_sql("select * from clean_exog_m")[EXOG].mean().to_dict()  # 역사적 평균=기준
    paths = {}
    for scn in SHOCK:
        la = lab_annual(scn)
        X = np.zeros((HMON, len(EXOG)))
        for t, d in enumerate(FUT):
            decay = max(1 - t / 36.0, 0.0)        # 36개월 선형 감쇠(일시적 충격)
            for j, v in enumerate(EXOG):
                x = base[v]
                if v == "lab_g":
                    x = la[d.year] / 12.0            # 연→월
                elif v in SHOCK[scn]:
                    x = base[v] + SHOCK[scn][v] * decay / 12.0
                X[t, j] = x
        paths[scn] = X
    return paths, list(EXOG)


def weights_2024():
    """70산업 2024 명목 부가가치 가중치(부모 명목 × 생산비중)."""
    imap = read_sql("select * from clean_industry_map")
    nom = read_sql("select quarter, code, value from clean_va_nom")
    nom24 = nom[nom["quarter"].str.startswith("2024")].groupby("code")["value"].sum()
    mfg = read_sql("select date,ksic,value from raw_mfg_prod")
    svc = read_sql("select date,ksic,value from raw_svc_prod")
    p24 = {}
    for src, df in [("mfg", mfg), ("svc", svc)]:
        d = df[df["date"].str.startswith("2024")].groupby("ksic")["value"].mean()
        for k, v in d.items():
            p24[(src, k)] = v
    w = {}
    for key, grp in imap.groupby("bok36"):
        codes = key.split(";")
        tot = float(nom24.reindex(codes).sum())
        rows = grp.to_dict("records")
        shares = []
        for r in rows:
            pv = p24.get((r["prod_src"], r["ksic"]), np.nan)
            shares.append(pv if pd.notna(pv) else np.nan)
        sh = np.array([s if pd.notna(s) else np.nanmean([x for x in shares if pd.notna(x)] or [1]) for s in shares])
        sh = sh / sh.sum() if sh.sum() > 0 else np.ones(len(rows)) / len(rows)
        for r, s in zip(rows, sh):
            w[r["ind_id"]] = tot * s
    s = pd.Series(w); return (s / s.sum())


def main():
    # VARX 재추정(점추정 + 잔차공분산)
    Y, ex, inds = v5.load()
    X, Yt, n, ne = v5.design(Y, ex)
    lam = v5.select_lambda(X, Yt)
    B = v5.ridge_fit(X, Yt, lam)
    resid = Yt - X @ B
    Sigma = np.cov(resid.T)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))

    paths, exn = exog_paths()
    w = weights_2024().reindex(inds).fillna(0).values

    yhist = Y.values[-P:]                 # 시드 내생(마지막 P개월 성장률)
    xhist = ex.values[-Q:]               # 시드 외생

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

    fan_rows = []; ind_rows = []
    aggfan = {}
    for scn in SHOCK:
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

    fan = pd.DataFrame(fan_rows, columns=["scenario", "date", "point", "p5", "p15", "p50", "p85", "p95"])
    ind = pd.DataFrame(ind_rows, columns=["scenario", "ind_id", "cum_growth_2035_pct"])
    to_sql(fan, "result_scenario_fan"); to_sql(ind, "result_scenario_industry_2035")
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
               for scn in SHOCK}
    print("시나리오 2035 총부가가치 지수(2025.12=100):")
    for scn in SHOCK:
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
