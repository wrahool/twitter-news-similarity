"""
16_postcount_trends.py -- H1 / common-trend models with post-count controls.

Trend estimates need the UNCENTRED SBERT (centring removes the common trend), so
scores come from the v2 averaged file (TF-IDF is identical in v2 and v3);
centred SBERT (v3) is added as `sbert_c` for the year-specific gradients.

Specs: no control / count controls (Nsum, Nprod as in 15_postcount_check.py) /
pair-months where both outlets have the full 500 posts.
Models:
  Model B      pair FE + beta*t + delta*Dt     (full window; excl. Mar 2020-Feb 2021;
                                               from Nov 2017 on; up to Oct 2017)
  year theta   pair + month FE, D x year (ref 2015)
  common trend yearly average of month effects (pair + month FE, Dt [+ controls])
Output: tables/postcount_trends.csv, tables/postcount_year_theta.csv,
        tables/postcount_common_trend.csv
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *

pc = importlib.import_module("15_postcount_check")
AVG_V2 = AVG_UNCENTRED                      # uncentred SBERT (see common.py)
AVG_V3 = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
CTRL = pc.CTRL
WINDOWS = {"full window": (None, None), "excl. COVID year": ("2020-03", "2021-02"),
           "Nov 2014 - Oct 2017": (None, "2017-10"), "Nov 2017 - Jan 2023": ("2017-11", None)}


def panel():
    common.DATA_SIM = AVG_V2
    p, _ = build_panel()
    common.DATA_SIM = AVG_V3
    q, _ = build_panel()
    p = p.merge(q[["month", "pair", "domain", "sbert"]].rename(columns={"sbert": "sbert_c"}),
                on=["month", "pair", "domain"], how="left")
    p = pc.prep(pc.add_counts(p))
    p["Dc"] = p.D - p.groupby("pair").D.first().mean()
    p["ybin"] = p.year.clip(2015, 2022)
    return p


def window(x, name):
    a, b = WINDOWS[name]
    if name == "excl. COVID year":
        return x[(x.month < a) | (x.month > b)]
    if a:
        x = x[x.month >= a]
    if b:
        x = x[x.month <= b]
    return x


SPECS = {"no control": ([], lambda x: x), "count controls": (CTRL, lambda x: x),
         "both outlets 500 posts": ([], lambda x: x[x.full])}

if __name__ == "__main__":
    p = panel()
    rows, theta, ct = [], [], []
    for dom in ["pol", "nonpol", "all"]:
        for spec, (ctrl, sub) in SPECS.items():
            if dom == "all" and spec != "no control":
                continue                      # 'all' always has 500 posts
            x = sub(p[p.domain == dom])
            for m in ["tfidf", "sbert"]:
                for w in WINDOWS:
                    r = fe_ols(window(x, w), m, ["t", "Dt"] + ctrl, ["pair"])
                    t = r["tab"]
                    rows.append(dict(domain=dom, measure=m, spec=spec, window=w,
                                     beta=t.loc["t", "coef"], se=t.loc["t", "se_dyadic"], p=t.loc["t", "p_dyadic"],
                                     delta=t.loc["Dt", "coef"], delta_p=t.loc["Dt", "p_dyadic"], n=r["n"]))
                # common trend: month effects net of pair FE (and controls)
                r = fe_ols(x, m, ["Dt"] + ctrl, ["pair", "month"])
                y = x[m].values - r["tab"].coef["Dt"] * x.Dt.values
                if ctrl:
                    y = y - sum(r["tab"].coef[c] * x[c].values for c in ctrl)
                me = pc.am.pair_effects(x.assign(yy=y).rename(columns={"pair": "month", "month": "pair"}), "yy")
                yr = pd.Series(me.values, index=pd.PeriodIndex(me.index, freq="M").year)
                for yv, v in yr.groupby(level=0).mean().items():
                    ct.append(dict(domain=dom, measure=m, spec=spec, year=yv, month_effect=v))
            for m in ["tfidf", "sbert", "sbert_c"]:
                years = sorted(y for y in x.ybin.unique() if y != 2015)
                xx = x.copy()
                for yv in years:
                    xx[f"Dy{yv}"] = xx.Dc * (xx.ybin == yv)
                r = fe_ols(xx, m, [f"Dy{yv}" for yv in years] + ctrl, ["pair", "month"])
                for yv in years:
                    b, se, pv = pc.coef(r, f"Dy{yv}")
                    theta.append(dict(domain=dom, measure=m, spec=spec, year=yv, theta=b, se=se, p=pv))
    rows, theta, ct = pd.DataFrame(rows), pd.DataFrame(theta), pd.DataFrame(ct)
    rows.to_csv(os.path.join(TAB, "postcount_trends.csv"), index=False)
    theta.to_csv(os.path.join(TAB, "postcount_year_theta.csv"), index=False)
    ct.to_csv(os.path.join(TAB, "postcount_common_trend.csv"), index=False)
    save_fallback_log("postcount_trends")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(rows.pivot_table(index=["domain", "measure", "window"], columns="spec", values=["beta", "p"]).round(4).to_string())
    print(theta[(theta.p < .1) | (theta.year.isin([2020, 2021]))].pivot_table(
        index=["domain", "measure", "year"], columns="spec", values=["theta", "p"]).round(4).to_string())
    print(ct.pivot_table(index=["domain", "measure", "year"], columns="spec", values="month_effect").round(4).to_string())
