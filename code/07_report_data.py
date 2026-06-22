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
    rep = ["H_ELEC", "MFG_AUTO", "FIN_BANK", "H_RETAIL", "CON", "AGR"]
    latent = {}
    for iid in rep:
        s = lv[lv.ind_id == iid].sort_values("date")
        v = s["va_level"].values
        latent[name.get(iid, iid)] = [round(x / v[0] * 100, 1) for x in v]
    dates = sorted(lv[lv.ind_id == rep[0]]["date"].unique())

    # 2) 진단
    diag = read_sql("select * from latent_diagnostics")
    try:
        hold = read_sql("select * from latent_holdout")
    except Exception:
        hold = pd.DataFrame(columns=["holdout_rmse", "holdout_corr"])
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
    # 4b) 시나리오 전년대비 증가율(YoY) 팬
    grow = read_sql("select * from result_scenario_growth")
    scen_growth = {}
    for scn in ["S1", "S2", "S3", "S4", "S5"]:
        d = grow[grow.scenario == scn].sort_values("year")
        scen_growth[scn] = dict(year=[str(y) for y in d["year"].tolist()],
                                point=d["g_point"].round(2).tolist(),
                                p5=d["g_p5"].round(2).tolist(),
                                p95=d["g_p95"].round(2).tolist())

    # 5) 산업별 2035 + 노동효과 상하위
    ind = read_sql("select * from result_scenario_industry_2035")
    piv = ind.pivot(index="ind_id", columns="scenario", values="cum_growth_2035_pct")
    piv["labor"] = piv["S3"] - piv["S2"]
    piv["name"] = [name.get(i, i) for i in piv.index]
    lab_top = piv.sort_values("labor", ascending=False).head(8)[["name", "S1", "S2", "S3", "labor"]].round(2).to_dict("records")
    lab_bot = piv.sort_values("labor").head(5)[["name", "S1", "S2", "S3", "labor"]].round(2).to_dict("records")

    # 6) 외생 동태승수(가중평균 부호)
    mult = read_sql("select * from result_varx_multipliers")

    # 7) 거시 블록(DSGE) 경로
    dm = read_sql("select * from result_dsge_macro")
    de = read_sql("select * from result_dsge_exog")
    dsge = {"quarters": sorted(dm[dm.scenario == "S1"]["quarter"].tolist()), "macro": {}, "exog": {}}
    for scn in ["S1", "S2", "S3", "S4", "S5"]:
        a = dm[dm.scenario == scn].sort_values("quarter")
        b = de[de.scenario == scn].sort_values("quarter")
        dsge["macro"][scn] = dict(Y_gap=a["Y_gap"].round(2).tolist(),
                                  inflation=a["inflation"].round(2).tolist(),
                                  policy_rate=a["policy_rate"].round(2).tolist(),
                                  unemp_gap=a["unemp_gap"].round(2).tolist())
        dsge["exog"][scn] = dict(lab_g=b["lab_g"].round(3).tolist())

    # 8) 신규 8개 산업(MECE carve-out) — 연평균 지수(2020=100) + 메타 (latent_monthly_va role='emerging')
    roles = imap.set_index("ind_id")["role"].to_dict()
    em = read_sql("select * from emerging_meta")
    base20e = read_sql("select ind_id, avg(va_level) b from latent_monthly_va where date>='2020-01-01' and date<='2020-12-01' group by ind_id").set_index("ind_id")["b"]
    elv = read_sql("select date, ind_id, va_level from latent_monthly_va")
    elv = elv[elv["ind_id"].isin(em["em_id"])].copy()
    elv["y"] = elv["date"].str[:4]
    nyears = sorted(elv["y"].unique())
    dgm = read_sql("select ind_id, mean_growth from latent_diagnostics").set_index("ind_id")["mean_growth"]
    nser = {}
    for eid, g in elv.groupby("ind_id"):
        yv = (g.sort_values("date").groupby("y")["va_level"].mean()) / base20e[eid] * 100
        arr = [round(float(v), 1) for v in yv.reindex(nyears).values]
        if eid == "E_PLAT":
            arr = [a if int(y) >= 2017 else None for a, y in zip(arr, nyears)]
        nser[eid] = arr
    new8 = {"years": nyears, "series": nser,
            "meta": [{"id": r["em_id"], "name": r["name"], "host": r["host_name"],
                      "share": round(float(r["share_2020_eff"]) * 100, 1),
                      "mean": round(float(dgm.get(r["em_id"], float("nan"))), 3)}
                     for _, r in em.iterrows()]}

    # 9) 70 월별 뷰어 데이터 → data/viewer_data.json
    base20 = base20e
    vlv = read_sql("select date, ind_id, va_level from latent_monthly_va")
    vdates = sorted(vlv["date"].unique())
    vser, vmeta = {}, []
    dg2 = read_sql("select ind_id, rho from latent_diagnostics").set_index("ind_id")["rho"]
    imap_i = imap.set_index("ind_id")
    for iid, g in vlv.groupby("ind_id"):
        gg = g.sort_values("date")
        vser[iid] = [round(float(x / base20[iid] * 100), 1) for x in gg["va_level"].values]
        role = roles.get(iid, "base")
        grp = "신산업" if role == "emerging" else imap_i.loc[iid, "sector"]
        vmeta.append({"id": iid, "name": imap_i.loc[iid, "ind_name"], "group": grp,
                      "rho": round(float(dg2.get(iid, float("nan"))), 2), "new": role == "emerging"})
    json.dump({"start": "2000-01", "n": len(vdates), "meta": vmeta, "series": vser},
              open(os.path.join(REPO, "data", "viewer_data.json"), "w", encoding="utf-8"), ensure_ascii=False)

    payload = dict(
        dates=dates, latent=latent, dsge=dsge, new8=new8,
        diag=dict(n=int(len(diag)), rho_mean=round(float(diag["rho"].mean()), 3),
                  recon_max=2e-16,
                  hold_rmse=round(float(hold["holdout_rmse"].mean()), 3) if len(hold) else None,
                  hold_corr=round(float(hold["holdout_corr"].mean()), 3) if len(hold) else None),
        varx=varx, conn_top=conn_top, conn_bot=conn_bot,
        scen_fan=scen_fan, scen_growth=scen_growth, lab_top=lab_top, lab_bot=lab_bot,
        sector_counts=imap["sector"].value_counts().to_dict(),
    )
    js = "window.CHART = " + json.dumps(payload, ensure_ascii=False) + ";"
    open(os.path.join(REPO, "chart_data.js"), "w", encoding="utf-8").write(js)
    print("chart_data.js written:", len(js), "bytes | dates", len(dates),
          "| 시나리오 2035", {k: scen_fan[k]['point'][-1] for k in scen_fan})


if __name__ == "__main__":
    main()
