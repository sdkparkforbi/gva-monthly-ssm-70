# -*- coding: utf-8 -*-
"""Phase E — 웹 보고서용 차트 데이터(chart_data.js) 생성."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _common import read_sql, ROOT
import os

REPO = ROOT


def main():
    imap = read_sql("select * from clean_industry_map")
    name = dict(zip(imap["ind_id"], imap["ind_name"]))

    # 1) 월별 잠재 부가가치(대표 6산업, 지수화 2000.01=100)
    lv = read_sql("select date, ind_id, va_level from latent_monthly_va")
    rep = ["MFG26", "MFG30", "FIN64", "WRT47", "CON", "AGR1"]
    latent = {}
    for iid in rep:
        s = lv[lv.ind_id == iid].sort_values("date")
        v = s["va_level"].values
        latent[name.get(iid, iid)] = [round(x / v[0] * 100, 1) for x in v]
    dates = sorted(lv[lv.ind_id == rep[0]]["date"].unique())

    # 2) 진단
    diag = read_sql("select * from latent_diagnostics")
    hold = read_sql("select * from latent_holdout")
    varx = read_sql("select * from result_varx_summary").iloc[0].to_dict()

    # 3) 연결성 상위/하위
    conn = read_sql("select * from result_connectedness")
    conn["name"] = conn["ind_id"].map(name)
    conn_top = conn.head(8)[["name", "FROM", "TO", "NET"]].round(1).to_dict("records")
    conn_bot = conn.tail(5)[["name", "FROM", "TO", "NET"]].round(1).to_dict("records")

    # 4) 시나리오 팬차트(총계, 연말)
    fan = read_sql("select * from result_scenario_fan")
    fan["year"] = fan["date"].str[:4]
    fy = fan[fan["date"].str.endswith("-12-01")].copy()
    scen_fan = {}
    for scn in ["S1", "S2", "S3", "S4", "S5"]:
        d = fy[fy.scenario == scn]
        scen_fan[scn] = dict(year=d["year"].tolist(),
                             point=d["point"].tolist(),
                             p5=d["p5"].tolist(), p95=d["p95"].tolist(),
                             p15=d["p15"].tolist(), p85=d["p85"].tolist())

    # 5) 산업별 2035 + 노동효과 상하위
    ind = read_sql("select * from result_scenario_industry_2035")
    piv = ind.pivot(index="ind_id", columns="scenario", values="cum_growth_2035_pct")
    piv["labor"] = piv["S3"] - piv["S2"]
    piv["name"] = [name.get(i, i) for i in piv.index]
    lab_top = piv.sort_values("labor", ascending=False).head(8)[["name", "S1", "S2", "S3", "labor"]].round(2).to_dict("records")
    lab_bot = piv.sort_values("labor").head(5)[["name", "S1", "S2", "S3", "labor"]].round(2).to_dict("records")

    # 6) 외생 동태승수(가중평균 부호)
    mult = read_sql("select * from result_varx_multipliers")

    payload = dict(
        dates=dates, latent=latent,
        diag=dict(n=int(len(diag)), rho_mean=round(float(diag["rho"].mean()), 3),
                  recon_max=float(diag["max_recon_err"].max()),
                  hold_rmse=round(float(hold["holdout_rmse"].mean()), 3),
                  hold_corr=round(float(hold["holdout_corr"].mean()), 3)),
        varx=varx, conn_top=conn_top, conn_bot=conn_bot,
        scen_fan=scen_fan, lab_top=lab_top, lab_bot=lab_bot,
        sector_counts=imap["sector"].value_counts().to_dict(),
    )
    js = "window.CHART = " + json.dumps(payload, ensure_ascii=False) + ";"
    open(os.path.join(REPO, "chart_data.js"), "w", encoding="utf-8").write(js)
    print("chart_data.js written:", len(js), "bytes | dates", len(dates),
          "| 시나리오 2035", {k: scen_fan[k]['point'][-1] for k in scen_fan})


if __name__ == "__main__":
    main()
