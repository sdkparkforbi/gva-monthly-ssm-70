# -*- coding: utf-8 -*-
"""거시 블록(DSGE) — BOKDPM 준구조 NK gap 모형의 파이썬 포트.

선행연구 Block 1을 본 후속연구로 이식. 전향 IS·하이브리드 필립스·테일러준칙·UIP·오쿤을
Fair–Taylor(Gauss–Seidel 시간반복)로 결정론적으로 푼다. 2026Q1~2035Q4(40분기=10년).

역할: 시나리오 충격(세계수요·유가·무역비용·국내수요·인구) → 거시 경로 + 산업모형 외생 5종.
산출: result_dsge_macro / result_dsge_exog (DB), data/dsge_*.csv
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import to_sql, DATA
import os

b1, b2, b3, b4, b6 = 0.50, 0.10, 0.10, 0.30, 0.10
lam1, lam2, lam3 = 0.55, 0.15, 0.65
g1, g2, g3 = 0.50, 1.50, 0.50
a1, a2 = 0.80, 0.30
om4, om7, om8 = 0.25, 0.50, 0.20
rrbar, pietar = 1.5, 2.0
H, BUF = 40, 24
N = H + BUF
SCN = ["S1", "S2", "S3", "S4", "S5"]
LAB = {"S1": "중위", "S2": "저위", "S3": "고위", "S4": "중위", "S5": "중위"}


def labor_force_growth():
    pop = pd.read_csv(os.path.join(DATA, "wap_projection.csv")).set_index("연도")
    col = {"중위": "중위_천명", "고위": "고위_천명", "저위": "저위_천명"}
    out = {}
    for a, c in col.items():
        s = pop[c].dropna()
        ann = {}
        for y in range(2026, 2036):
            if y in s.index and (y - 1) in s.index:
                ann[y] = np.log(s.loc[y] / s.loc[y - 1]) * 100
            else:
                ann[y] = ann.get(y - 1, list(ann.values())[-1] if ann else -1.0)
        out[a] = np.array([ann[2026 + h // 4] / 4 for h in range(H)])
    return out


def scenario_inputs(s):
    z = np.zeros(N)
    dec = lambda k: np.clip(1 - np.arange(N) / k, 0, 1)
    yus = z.copy(); oil = z.copy(); tcost = z.copy(); dshk = z.copy()
    if s == "S2":
        yus = -1.2 * dec(12); oil = -8.0 * dec(8); tcost = +1.5 * dec(12); dshk = -1.0 * dec(12)
    if s == "S4":
        yus = +0.8 * dec(20); dshk = +0.6 * dec(28)
    if s == "S5":
        oil = +15.0 * dec(8); tcost = +2.0 * dec(10); yus = -0.6 * dec(10); dshk = -0.8 * dec(10)
    return yus, oil, tcost, dshk


def solve_korea_block(yus, oil, dshk, iters=400):
    Y = np.zeros(N); PIE = np.full(N, pietar); RS = np.full(N, rrbar + pietar); S = np.zeros(N)
    for _ in range(iters):
        Y0, RS0, S0 = Y.copy(), RS.copy(), S.copy()
        for t in range(N):
            yl = Y[t - 1] if t > 0 else 0.0
            yf = Y[t + 1] if t + 1 < N else 0.0
            rsl = RS[t - 1] if t > 0 else rrbar + pietar
            rgap_l = rsl - pietar - rrbar
            Y[t] = b1 * yl + b2 * yf - b3 * rgap_l + b4 * yus[t] - b6 * oil[t] / 10 + dshk[t]
            pif = PIE[t + 1] if t + 1 < N else pietar
            pil = PIE[t - 1] if t > 0 else pietar
            PIE[t] = lam3 * pietar + (1 - lam3) * (lam1 * pif + (1 - lam1) * pil + lam2 * 4 * yl)
            pie_e = PIE[t + 1] if t + 1 < N else pietar
            RS[t] = g1 * rsl + (1 - g1) * (rrbar + pietar + g2 * (pie_e - pietar) + g3 * 4 * Y[t])
            sf = S[t + 1] if t + 1 < N else 0.0
            sl = S[t - 1] if t > 0 else 0.0
            rgap = RS[t] - pietar - rrbar
            S[t] = (1 - 0.1) * (om7 * sf + (1 - om7) * sl - om4 * rgap) + om8 * (-Y[t])
        if max(np.abs(Y - Y0).max(), np.abs(RS - RS0).max(), np.abs(S - S0).max()) < 1e-8:
            break
    RR = RS - PIE
    return Y[:H], PIE[:H], RS[:H], RR[:H], S[:H]


def main():
    lfg = labor_force_growth()
    partic = np.linspace(0.05, 0.0, H)
    quarters = [f"{2026+h//4}Q{h%4+1}" for h in range(H)]
    rows, macro = [], []
    for s in SCN:
        yus, oil_sh, tcost, dshk = scenario_inputs(s)
        Y, PIE, RS, RR, S = solve_korea_block(yus, oil_sh, dshk)
        UNRgap = np.zeros(H)
        for t in range(H):
            UNRgap[t] = a1 * (UNRgap[t - 1] if t > 0 else 0) - a2 * Y[t]
        dUNR = np.diff(np.concatenate([[0], UNRgap]))
        oil_g = 0.5 + oil_sh[:H] / 8 * 4
        trd_g = 1.0 + 0.8 * yus[:H] - 0.6 * tcost[:H]
        d_rrate = np.diff(np.concatenate([[RR[0]], RR]))
        rfx_g = np.diff(np.concatenate([[0], S]))
        lab_g = lfg[LAB[s]] + partic - dUNR
        for h in range(H):
            rows.append([s, quarters[h], round(oil_g[h], 3), round(trd_g[h], 3),
                         round(d_rrate[h], 3), round(rfx_g[h], 3), round(lab_g[h], 3)])
            macro.append([s, quarters[h], round(Y[h], 3), round(PIE[h], 3), round(RS[h], 3),
                          round(RR[h], 3), round(S[h], 3), round(UNRgap[h], 3)])
    ex = pd.DataFrame(rows, columns=["scenario", "quarter", "oil_g", "trd_g", "d_rrate", "rfx_g", "lab_g"])
    mc = pd.DataFrame(macro, columns=["scenario", "quarter", "Y_gap", "inflation", "policy_rate",
                                      "real_rate", "fx_gap", "unemp_gap"])
    to_sql(ex, "result_dsge_exog"); to_sql(mc, "result_dsge_macro")
    ex.to_csv(os.path.join(DATA, "dsge_exog_paths.csv"), index=False, encoding="utf-8-sig")
    mc.to_csv(os.path.join(DATA, "dsge_macro_paths.csv"), index=False, encoding="utf-8-sig")
    print("거시 블록(DSGE) 완료 | 2026Q1~2035Q4 40분기 ×", len(SCN), "시나리오")
    i30 = (2030 - 2026) * 4
    print(f"{'scn':<4}{'2026 산출갭':>11}{'2030 산출갭':>11}{'2030 실업갭':>11}{'10년 고용Σ':>11}")
    for s in SCN:
        d = mc[mc.scenario == s].reset_index(drop=True)
        labsum = ex[ex.scenario == s]["lab_g"].sum()
        print(f"{s:<4}{d.loc[0,'Y_gap']:>11.2f}{d.loc[i30,'Y_gap']:>11.2f}{d.loc[i30,'unemp_gap']:>11.2f}{labsum:>11.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
