# -*- coding: utf-8 -*-
"""3개 노트북(.ipynb) 생성: 데이터 / 상태공간 / VARX·시나리오."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def md(*t): return {"cell_type": "markdown", "metadata": {}, "source": list(t)}
def code(*t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": list(t)}


N1 = nb([
 md("# GVA70 — ① 데이터 입수·정제 (KOSIS·한국은행 → SQLite)\n",
    "혼합주기 상태공간 모형의 입력을 단일 SQLite DB로 적재한다.\n",
    "- 헤드라인 프록시 **실질주가가치** = 코스피 산업별 주가지수 ÷ CPI (KOSIS 한국거래소 + 한국은행)\n",
    "- 산업별 광공업·서비스업 생산지수, 36산업 분기 실질부가가치, 월별 외생\n"),
 code("import sys; sys.path.insert(0,'.')\n", "import importlib\n",
      "for m in ['01_fetch_raw','01b_fetch_fix','00_industry_map','03_clean_build']:\n",
      "    importlib.import_module(m).main() if hasattr(importlib.import_module(m),'main') else None"),
 md("### 적재 확인"),
 code("from _common import read_sql\n",
      "for t in ['raw_kospi_sector','raw_mfg_prod','raw_svc_prod','raw_va_q','clean_exog_m','clean_industry_map']:\n",
      "    print(t, read_sql(f'select count(*) c from {t}').iloc[0,0])"),
])

N2 = nb([
 md("# GVA70 — ② 혼합주기 상태공간 → 월별 잠재 부가가치\n",
    "관측식(실질주가가치·생산지수·외생)과 상태식(월별 실질부가가치 잠재변수),\n",
    "분기=3개월 평균 시점집계 제약을 누적기로 증강하고 자체 칼만필터/평활로 분해한다.\n",
    "$$m_{i,t}=\\beta_i'X_{i,t}+u_{i,t},\\; u_{i,t}=\\rho_i u_{i,t-1}+\\eta_{i,t},\\quad Y^Q_{i,\\tau}=\\tfrac13\\sum_{k} m_{i,k}$$"),
 code("import sys; sys.path.insert(0,'.')\n", "import importlib\n",
      "importlib.import_module('04_mf_statespace').main()"),
 md("### 진단(정합성·holdout)"),
 code("from _common import read_sql\n",
      "print(read_sql('select * from latent_diagnostics').describe(numeric_only=True))\n",
      "print(read_sql('select avg(holdout_rmse) rmse, avg(holdout_corr) corr from latent_holdout'))"),
])

N3 = nb([
 md("# GVA70 — ③ VARX 추정 · 시나리오 전망\n",
    "월별 잠재 부가가치 70산업으로 빈도주의 정칙화 VARX를 추정하고,\n",
    "노동공급(중위·저위·고위)×대외환경 다섯 시나리오로 2035년까지 전망한다."),
 code("import sys; sys.path.insert(0,'.')\n", "import importlib\n",
      "importlib.import_module('05_varx_estimate').main()\n",
      "importlib.import_module('06_scenario').main()\n",
      "importlib.import_module('07_report_data').main()"),
 md("### 결과 요약"),
 code("from _common import read_sql\n",
      "print(read_sql('select * from result_varx_summary'))\n",
      "print(read_sql('select * from result_scenario_fan where date=\"2035-12-01\"'))"),
])

for fn, obj in [("GVA70_1_데이터.ipynb", N1), ("GVA70_2_상태공간.ipynb", N2),
                ("GVA70_3_VARX시나리오.ipynb", N3)]:
    p = os.path.join(ROOT, "code", fn)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", fn)
