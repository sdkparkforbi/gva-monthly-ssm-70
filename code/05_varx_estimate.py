# -*- coding: utf-8 -*-
"""Phase C — 월별 빈도주의 VARX 추정(정칙화 OLS = 릿지).

  y_t (70산업 월별 실질부가가치 성장률) = c + Σ_{l=1}^p B_l y_{t-l} + Σ_{s=0}^q Γ_s x_{t-s} + u_t
  x_t = 월별 외생[oil_g, trd_g, d_rrate, rfx_g, gpr_l]

선행연구의 베이지안 BVARX와 달리, 70차원 다수모수를 릿지(λ는 보류표본 MSE로 선택)로 안정 추정.
산출: 계수, 잔차공분산, 안정성, 외생 동태승수, Diebold–Yılmaz 연결성(GFEVD).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import read_sql, to_sql, DATA
import os

P, Q = 2, 2          # 내생/외생 시차
EXOG = ["oil_g", "trd_g", "d_rrate", "rfx_g", "gpr_l", "lab_g"]
H_FEVD = 24          # 연결성 지평(월)


def load():
    lv = read_sql("select date, ind_id, va_growth from latent_monthly_va")
    Y = lv.pivot(index="date", columns="ind_id", values="va_growth").sort_index()
    inds = list(Y.columns)
    ex = read_sql("select * from clean_exog_m").set_index("date")[EXOG]
    ex = ex.reindex(Y.index)
    # 결측 보간(외생 일부 초기 결측)
    ex = ex.interpolate(limit_direction="both")
    Y = Y.iloc[1:]          # 첫 행 성장률 NaN 제거
    ex = ex.loc[Y.index]
    return Y, ex, inds


def design(Y, ex):
    Yv, Xe = Y.values, ex.values
    Tt, n = Yv.shape
    ne = Xe.shape[1]
    start = P
    rows_y, rows_x = [], []
    for t in range(start, Tt):
        lagsy = np.concatenate([Yv[t - l] for l in range(1, P + 1)])
        lagsx = np.concatenate([Xe[t - s] for s in range(0, Q + 1)])
        rows_x.append(np.concatenate([[1.0], lagsy, lagsx]))
        rows_y.append(Yv[t])
    X = np.array(rows_x); Yt = np.array(rows_y)
    return X, Yt, n, ne


def ridge_fit(X, Yt, lam, n_endog=None):
    """릿지. 절편·외생블록은 비패널티(내생 동학만 축소, Minnesota 취지)."""
    k = X.shape[1]
    pen = np.eye(k); pen[0, 0] = 0.0       # 절편 비패널티
    if n_endog is not None:
        for c in range(1 + n_endog * P, k):   # 외생블록 비패널티
            pen[c, c] = 0.0
    B = np.linalg.solve(X.T @ X + lam * pen, X.T @ Yt)
    return B


def select_lambda(X, Yt, n_endog=None):
    """마지막 24개월 보류표본 1-step MSE로 λ 선택."""
    grid = [1, 3, 10, 30, 100, 300, 1000]
    h = 24
    Xtr, Ytr, Xte, Yte = X[:-h], Yt[:-h], X[-h:], Yt[-h:]
    best = None
    for lam in grid:
        B = ridge_fit(Xtr, Ytr, lam, n_endog)
        mse = float(np.mean((Xte @ B - Yte) ** 2))
        if best is None or mse < best[1]:
            best = (lam, mse)
    return best[0]


def companion(B, n):
    """내생계수만으로 동반행렬(stacked) 구성 → max|eig|."""
    A = [B[1 + l * n:1 + (l + 1) * n].T for l in range(P)]   # 각 (n x n)
    top = np.hstack(A)
    comp = np.zeros((n * P, n * P))
    comp[:n] = top
    if P > 1:
        comp[n:, :n * (P - 1)] = np.eye(n * (P - 1))
    return comp, A


def girf_connectedness(A, Sigma, n, H=H_FEVD):
    """일반화 FEVD(Pesaran–Shin) → Diebold–Yılmaz 연결성."""
    # MA(∞) Θ_h
    Theta = [np.eye(n)]
    comp = np.zeros((n * P, n * P)); comp[:n] = np.hstack(A)
    if P > 1:
        comp[n:, :n * (P - 1)] = np.eye(n * (P - 1))
    Cp = np.eye(n * P)
    for h in range(1, H):
        Cp = comp @ Cp
        Theta.append(Cp[:n, :n])
    sig = np.diag(Sigma).copy(); sig[sig <= 0] = 1e-8
    denom = np.zeros(n); num = np.zeros((n, n))
    for Th in Theta:
        TS = Th @ Sigma
        denom += np.einsum("ij,ij->i", Th @ Sigma, Th)
        num += (TS ** 2) / sig[None, :]
    gfevd = num / denom[:, None]
    gfevd = gfevd / gfevd.sum(axis=1, keepdims=True)   # 행 정규화
    frm = (gfevd.sum(axis=1) - np.diag(gfevd)) * 100 / n  # 평균 기여(타→i)
    to = (gfevd.sum(axis=0) - np.diag(gfevd)) * 100 / n
    net = to - frm
    total = (gfevd.sum() - np.trace(gfevd)) * 100 / n
    return gfevd, frm, to, net, total


def exog_multipliers(A, B, n, ne, H=24):
    """외생 1단위 충격에 대한 누적 동태승수(레벨, 20개월)."""
    # Γ_0 = 외생 동시계수 (X열에서 내생시차 다음 블록의 첫 ne개)
    g0 = B[1 + n * P:1 + n * P + ne].T            # n x ne
    D = [g0.copy()]
    Dprev = [np.zeros((n, ne))] * P
    Dcur = g0.copy()
    hist = [Dcur.copy()]
    for h in range(1, H):
        acc = np.zeros((n, ne))
        for l in range(1, P + 1):
            if h - l >= 0:
                acc += A[l - 1] @ hist[h - l]
        hist.append(acc)
    cum = np.cumsum(np.array(hist), axis=0)[-1]    # 누적 레벨반응
    return cum                                      # n x ne


def main():
    Y, ex, inds = load()
    X, Yt, n, ne = design(Y, ex)
    print(f"표본 T={Yt.shape[0]} | 내생 n={n} | 외생 ne={ne} | 회귀변수 k={X.shape[1]}")
    lam = select_lambda(X, Yt, n)
    B = ridge_fit(X, Yt, lam, n)
    resid = Yt - X @ B
    Sigma = np.cov(resid.T)
    comp, A = companion(B, n)
    eig = np.max(np.abs(np.linalg.eigvals(comp)))
    r2 = 1 - resid.var(axis=0) / Yt.var(axis=0)
    print(f"λ*={lam} | max|eig|={eig:.3f} | 평균 R²={r2.mean():.3f}")

    # 저장: 계수
    rownames = ["const"] + [f"L{l}.{inds[i]}" for l in range(1, P + 1) for i in range(n)] + \
               [f"X{s}.{EXOG[j]}" for s in range(Q + 1) for j in range(ne)]
    coef = pd.DataFrame(B, index=rownames, columns=inds)
    to_sql(coef.reset_index().rename(columns={"index": "term"}), "result_varx_coef")
    pd.DataFrame(Sigma, index=inds, columns=inds).to_csv(
        os.path.join(DATA, "varx_sigma.csv"), encoding="utf-8-sig")

    # 외생 동태승수
    cum = exog_multipliers(A, B, n, ne)
    mult = pd.DataFrame(cum, index=inds, columns=EXOG)
    to_sql(mult.reset_index().rename(columns={"index": "ind_id"}), "result_varx_multipliers")
    mult.to_csv(os.path.join(DATA, "varx_exog_multipliers.csv"), encoding="utf-8-sig")

    # 연결성
    gfevd, frm, to, net, total = girf_connectedness(A, Sigma, n)
    conn = pd.DataFrame({"ind_id": inds, "FROM": frm, "TO": to, "NET": net}).sort_values("NET", ascending=False)
    to_sql(conn, "result_connectedness")
    conn.to_csv(os.path.join(DATA, "connectedness.csv"), index=False, encoding="utf-8-sig")

    summ = pd.DataFrame([{
        "T": Yt.shape[0], "n": n, "p": P, "q": Q, "lambda": lam,
        "max_eig": round(float(eig), 4), "mean_R2": round(float(r2.mean()), 4),
        "total_connectedness": round(float(total), 2)}])
    to_sql(summ, "result_varx_summary")
    summ.to_csv(os.path.join(DATA, "varx_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"총연결성(TCI)={total:.1f}% | NET 상위: {conn.head(3)['ind_id'].tolist()} | 하위: {conn.tail(3)['ind_id'].tolist()}")
    print("DONE")


if __name__ == "__main__":
    main()
