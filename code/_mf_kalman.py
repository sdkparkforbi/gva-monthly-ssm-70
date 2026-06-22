# -*- coding: utf-8 -*-
"""혼합주기 상태공간 — 커스텀 칼만필터/평활(temporal disaggregation).

선행연구(statsmodels DynamicFactor)와 구분되는 자체 구현.

모형(산업 i, 월 t):
  관측(고빈도, 매월) :  z_{j,t} = a_j + b_j · m_t + e_{j,t},   e ~ N(0, h_j)
       z = 프록시[실질주가가치(로그), 생산지수(로그), …] + 월별 외생은 회귀항 X로 흡수
  상태(월별 잠재)    :  m_t = (월별 실질부가가치 로그수준) = β'X_t + u_t,  u_t = ρ u_{t-1} + η_t
  관측(저빈도, 분기) :  분기 실질부가가치(로그수준) Q_τ = (1/3)Σ_{k∈τ} m_k   (시점집계 제약; 누적기 상태)

상태벡터 α_t = [u_t, C_t]'  (C_t = 분기 내 m 누적/3, 분기말 리셋).
표준 Chow–Lin/Litterman 시점분해를 누적기(Harvey cumulator) 상태공간으로 구현하고
자체 칼만필터로 우도(ρ 프로파일 격자탐색)·평활(월별 m_t 추출)을 수행한다.
"""
import numpy as np


def _build_X(months, indicators, exog):
    """회귀행렬 X_t = [const, trend, 프록시(로그표준화), 외생]. (T x k)"""
    T = len(months)
    cols = [np.ones(T), np.linspace(0, 1, T)]
    names = ["const", "trend"]
    for nm, v in indicators.items():
        cols.append(v); names.append(nm)
    for nm, v in exog.items():
        cols.append(v); names.append(nm)
    return np.column_stack(cols), names


def chowlin_kalman(q_level, q_month_idx, X, rho_grid=None):
    """누적기 상태공간 + 칼만필터/평활로 월별 잠재 수준 m_t 추출 (플로 = 합 집계).

    부가가치는 플로이므로 분기치 = 3개월 '합'(평균 아님): Y^Q_τ = Σ_{k∈τ} m_k.
    q_level    : 분기 수준 관측(분기 합) (len Q)
    q_month_idx: 각 분기의 '마지막 월' 인덱스(0-based, 월배열 기준) (len Q)
    X          : (T x k) 회귀행렬(월)
    반환: m_hat(T, 월수준), beta(k,), rho, loglik, resid_var
    """
    T, k = X.shape
    if rho_grid is None:
        rho_grid = np.r_[np.linspace(0.0, 0.95, 20), 0.97, 0.99]

    # 분기 집계행렬 A: (Q x T), 각 분기 = 직전 3개월의 '합'(플로, 1.0)
    Q = len(q_level)
    A = np.zeros((Q, T))
    for i, me in enumerate(q_month_idx):
        for k3 in range(3):
            j = me - k3
            if 0 <= j < T:
                A[i, j] = 1.0

    yq = np.asarray(q_level, float)
    valid = ~np.isnan(yq)
    A = A[valid]; yq = yq[valid]; Q = len(yq)

    best = None
    for rho in rho_grid:
        # u_t = rho u_{t-1}+eta, Var(u)=sigma^2/(1-rho^2). 월 공분산 Vu (T x T)
        if rho < 1:
            lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
            Vu = rho ** lag / (1 - rho ** 2 + 1e-12)
        else:
            Vu = np.minimum.outer(np.arange(1, T + 1), np.arange(1, T + 1)).astype(float)
        # 분기수준 GLS: yq = (A X) beta + A u ;  Cov(A u) = A Vu A' * s2
        AX = A @ X
        Omega = A @ Vu @ A.T
        Omega += 1e-8 * np.eye(Q)
        Oi = np.linalg.pinv(Omega)
        beta = np.linalg.solve(AX.T @ Oi @ AX + 1e-8 * np.eye(k), AX.T @ Oi @ yq)
        r = yq - AX @ beta
        s2 = float(r @ Oi @ r) / max(Q - k, 1)
        # 집중우도
        sign, logdet = np.linalg.slogdet(Omega * s2)
        ll = -0.5 * (Q * np.log(2 * np.pi) + logdet + Q)
        if best is None or ll > best[0]:
            # BLUE 분포: u_hat = Vu A' Omega^{-1} r  (Chow–Lin)
            u_hat = Vu @ A.T @ Oi @ r
            m_hat = X @ beta + u_hat
            best = (ll, rho, beta, m_hat, s2)
    ll, rho, beta, m_hat, s2 = best
    return dict(m=m_hat, beta=beta, rho=rho, loglik=ll, s2=s2)


def zscore_log(v):
    """양수 시계열 → 로그 후 표준화(결측 보존)."""
    v = np.asarray(v, float)
    out = np.full_like(v, np.nan)
    m = v > 0
    lg = np.log(v[m])
    out[m] = (lg - lg.mean()) / (lg.std() + 1e-9)
    return out
