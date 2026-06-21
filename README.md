# 혼합주기 상태공간 × 70개 산업 월별 실질부가가치 — VARX 시나리오 전망 (2026~2035)

> 실질주가가치·산업생산지수를 **관측방정식**에, 월별 실질부가가치를 **잠재상태**에 둔
> 혼합주기 상태공간 모형으로 70개 산업의 월별 부가가치를 복원하고, 그 월별 시계열로 VARX를 추정해 전망한다.
> 선행연구 [`gva-scenario-2025-2035`](https://github.com/sdkparkforbi/gva-scenario-2025-2035)의 후속(별도 연구).

**📄 문서 보기:** https://sdkparkforbi.github.io/gva-monthly-ssm-70/

## 무엇을 했나

1. **데이터 (Phase A)** — 통계청 KOSIS·한국은행 ECOS에서 월별 자료를 입수해 단일 **SQLite** DB로 적재·정제.
   - 헤드라인 프록시 **실질주가가치** = 코스피 산업별 주가지수(25섹터) ÷ CPI
   - 산업별 광공업·서비스업 생산지수, 36산업 분기 실질부가가치, 월별 외생(유가·교역·금리·환율·통화·취업자·GPR)
2. **혼합주기 상태공간 (Phase B)** — 자체 칼만필터/평활로 분기 부가가치를 월로 분해.
   - 관측식: 실질주가가치·생산지수·외생 / 상태식: 월별 실질부가가치 잠재변수 / 분기=3개월 평균 제약(누적기)
   - 2000.01~2025.12 **70개 산업 × 312개월** 월별 잠재 부가가치 복원 (분기 정합성 재구성오차 ≈ 0)
3. **VARX (Phase C)** — 월별 잠재 부가가치 70산업 + 외생의 **빈도주의 정칙화 VARX**.
   - 안정성(동반행렬 max|eig|<1)·Diebold–Yılmaz 연결성·외생 동태승수
4. **시나리오 (Phase D)** — 선행연구와 동일 설계(노동공급 중위/저위/고위 × 대외환경)로 S1~S5 월별 전망 + 몬테카를로 팬차트.
5. **보고서 (Phase E)** — 웹(GitHub Pages)·PDF(Playwright)·GitHub·Google Drive.

## 선행연구와의 차별 (다른 repo·DB·프로그램)

| | 선행연구 | 본 후속연구 |
|---|---|---|
| 산업 | 36개 · 분기 | **70개 · 월별** |
| 핵심 모형 | 베이지안 VARX + 준구조 거시 | **혼합주기 상태공간(잠재 월별 부가가치) → 빈도주의 VARX** |
| 추정 | 베이지안(미네소타), DynamicFactor | **자체 칼만필터 + 릿지 OLS** |
| 자료 저장 | 산재한 CSV | **단일 SQLite DB** |
| 저장소 | gva-scenario-2025-2035 | **gva-monthly-ssm-70** |

## 구조

```
code/
  _common.py            DB·ECOS·KOSIS·FRED 공통 유틸
  _mf_kalman.py         혼합주기 누적기 칼만필터/평활(Chow–Lin 일반화)
  00_industry_map.py    70개 산업분류 매핑(KSIC↔BOK36↔코스피섹터)
  01_fetch_raw.py / 01b_fetch_fix.py   원자료 수집 → raw_*
  03_clean_build.py     정제·정상성변환 → clean_*
  04_mf_statespace.py   월별 잠재 부가가치 추출 → latent_*
  05_varx_estimate.py   빈도주의 VARX → result_varx_*
  06_scenario.py        S1~S5 시나리오·팬차트 → result_scenario_*
  07_report_data.py     웹 보고서용 chart_data.js
  make_pdf.py           index.html → PDF
  GVA70_1_데이터.ipynb / _2_상태공간.ipynb / _3_VARX시나리오.ipynb
db/gva_monthly.sqlite   단일 DB (raw_* / clean_* / latent_* / result_*)
data/                   주요 결과 CSV · 산업분류 · 차트데이터
index.html              웹 보고서
gva-monthly-ssm-70-paper.pdf
```

## 재현

```bash
pip install -r requirements.txt
python code/01_fetch_raw.py        # 원자료 수집(API)
python code/01b_fetch_fix.py       # 서비스생산·분기VA·GPR 보완
python code/00_industry_map.py     # 70개 산업 매핑
python code/03_clean_build.py      # 정제·변환
python code/04_mf_statespace.py    # 혼합주기 상태공간 → 월별 잠재 부가가치
python code/05_varx_estimate.py    # VARX
python code/06_scenario.py         # 시나리오·팬차트
python code/07_report_data.py      # 보고서 데이터
python code/make_pdf.py            # PDF
```
API 키(`API_KEY_BOK.txt`, `API_KEY_KOSIS.txt`)는 저장소에 포함하지 않는다.

## 한계
주가-부가가치 매칭(25섹터→70산업 중 54개)·70개 분기 벤치마크 부재(생산지수 비중 배분)·
외생충격 축약형 부호·연쇄가중 실질 비가법성·월별 지표 없는 산업(농림어업·공공행정). 문서 §5 참조.

---
*정식 논문 집필 전 팀 내부 공유용 초안. 모든 수치는 위 코드로 재현 가능합니다.*
