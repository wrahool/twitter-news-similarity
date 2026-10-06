"""
15_postcount_check.py -- is centred SBERT (and TF-IDF) driven by how many posts
sit behind each outlet-month's score?

Post counts come from auxiliary/sampling_audit.csv (one row per outlet x month x
condition; S1_N..S5_N are identical, = min(500, available)). The "all" condition
always has 500 posts, so only political / non-political scores can be affected.

Count terms (per pair-month), with l_i = log(N_i / 500) <= 0:
  Nsum  = l_1 + l_2        (both outlets' sample sizes)
  Nprod = l_1 * l_2        (both thin at once)
  thin1 / thin2 dummies: exactly one / both outlets below 500 posts.

Checks, on the scores averaged over the five samples:
  A. mechanical dependence: y ~ count terms | month FE, and | pair + month FE
  B. key estimates without / with count controls / on full-500 pair-months only:
       delta (D x t), H3 contrast (D x t x Pol), right-right trend (RR x t),
       M x t, and the level gradient on month-adjusted pair means
  for Pew and Ad Fontes slant.
Output: tables/postcount_mechanical.csv, tables/postcount_robustness.csv
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *

am = importlib.import_module("13_adjusted_means")
fl = importlib.import_module("07_first_last")
tr = importlib.import_module("10_type_regression")
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
AUDIT = os.path.join(ROOT, "auxiliary", "sampling_audit.csv")
COND = {"pol": "political", "nonpol": "non_political", "all": "all"}
CTRL = ["Nsum", "Nprod"]


def add_counts(p):
    a = pd.read_csv(AUDIT)
    a["dom"] = a.Condition.map({v: k for k, v in COND.items()})
    n = a.set_index(["dom", "Month", "Outlet"]).S1_N
    p = p.copy()
    p["N1"] = n.reindex(pd.MultiIndex.from_arrays([p.domain, p.month, p.o1])).values
    p["N2"] = n.reindex(pd.MultiIndex.from_arrays([p.domain, p.month, p.o2])).values
    assert p[["N1", "N2"]].notna().all().all(), "missing post counts"
    l1, l2 = np.log(p.N1 / 500), np.log(p.N2 / 500)
    p["Nsum"], p["Nprod"] = l1 + l2, l1 * l2
    k = (p.N1 < 500).astype(int) + (p.N2 < 500).astype(int)
    p["thin1"], p["thin2"] = (k == 1).astype(int), (k == 2).astype(int)
    p["full"] = (k == 0)
    return p


def prep(p):
    p = p.copy()
    pm = p.groupby("pair").D.first().mean()
    p["Dt"] = (p.D - pm) * p.t
    p["Mt"] = (p.M - p.groupby("pair").M.first().mean()) * p.t
    p["RRt"] = (p.bloc_type == "RR") * p.t
    p["LRt"] = (p.bloc_type == "LR") * p.t
    return p


def coef(r, v):
    t = r["tab"]
    return t.loc[v, "coef"], t.loc[v, "se_dyadic"], t.loc[v, "p_dyadic"]


def estimates(p, controls, label, src):
    rows = []
    def add(fam, dom, m, v, r):
        b, se, pv = coef(r, v)
        rows.append(dict(source=src, spec=label, family=fam, domain=dom, measure=m, est=b, se=se, p=pv, n=r["n"]))
    for dom in ["pol", "nonpol"]:
        x = p[p.domain == dom]
        for m in ["sbert", "tfidf"]:
            add("delta D x t", dom, m, "Dt", fe_ols(x, m, ["Dt"] + controls, ["pair", "month"]))
            add("M x t", dom, m, "Mt", fe_ols(x, m, ["Dt", "Mt"] + controls, ["pair", "month"]))
            add("RR x t (vs LL)", dom, m, "RRt", fe_ols(x, m, ["LRt", "RRt"] + controls, ["pair", "month"]))
            # level gradient: pair effects net of month FE (and of count terms)
            y = x[m].values
            if controls:
                r = fe_ols(x, m, controls, ["pair", "month"])
                g = r["tab"].coef
                y = y - sum(g[c] * x[c].values for c in controls)
            xx = x.assign(yy=y)
            a = am.pair_effects(xx, "yy")
            keys = xx.groupby("pair")[["o1", "o2", "D"]].first()
            keys["adj"] = a.reindex(keys.index)
            keys = keys.reset_index()
            b, se, pv, _ = fl.ols_dyadic(keys, "adj")
            rows.append(dict(source=src, spec=label, family="level gradient (pair means on D)", domain=dom, measure=m,
                             est=b[1], se=se[1], p=pv[1], n=len(keys),
                             std_beta=b[1] * keys.D.std() / keys.adj.std()))
            f = tr.fit(keys.rename(columns={"adj": "y"}), "y")
            fr = f[f.term.str.startswith("Ideological")].iloc[0]
            rows.append(dict(source=src, spec=label, family="level gradient + format/UK", domain=dom, measure=m,
                             est=fr.est, se=fr.se, p=fr.p, n=len(keys)))
    # H3 stacked
    s = p[p.domain.isin(["pol", "nonpol"])].copy()
    s["pol"] = (s.domain == "pol").astype(int)
    s["Dt_pol"] = s.Dt * s.pol
    s["pair_dom"], s["month_dom"] = s.pair + "#" + s.domain, s.month + "#" + s.domain
    cx = []
    for c in controls:
        s[c + "_pol"] = s[c] * s.pol
        cx += [c, c + "_pol"]
    for m in ["sbert", "tfidf"]:
        add("H3 D x t x Pol", "pol-nonpol", m, "Dt_pol",
            fe_ols(s, m, ["Dt", "Dt_pol"] + cx, ["pair_dom", "month_dom"]))
    return rows


if __name__ == "__main__":
    common.DATA_SIM = AVG
    mech, rob = [], []
    for src in ["pew", "adfontes"]:
        common.SLANT_SOURCE = src
        p, _ = build_panel()
        p = prep(add_counts(p))
        if src == "pew":
            for dom in ["pol", "nonpol"]:
                x = p[p.domain == dom]
                for m in ["sbert", "tfidf"]:
                    for fe in (["month"], ["pair", "month"]):
                        for xv in (CTRL, ["thin1", "thin2"]):
                            r = fe_ols(x, m, xv, fe)
                            for v in xv:
                                b, se, pv = coef(r, v)
                                mech.append(dict(domain=dom, measure=m, fe=" + ".join(fe), term=v, est=b, se=se, p=pv,
                                                 share_thin_rows=1 - x.full.mean()))
        rob += estimates(p, [], "no control", src)
        rob += estimates(p, CTRL, "count controls", src)
        rob += estimates(p[p.full], [], "both outlets 500 posts", src)
    mech, rob = pd.DataFrame(mech), pd.DataFrame(rob)
    mech.to_csv(os.path.join(TAB, "postcount_mechanical.csv"), index=False)
    rob.to_csv(os.path.join(TAB, "postcount_robustness.csv"), index=False)
    save_fallback_log("postcount")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(mech.round(4).to_string(index=False))
    w = rob.pivot_table(index=["source", "family", "domain", "measure"], columns="spec", values=["est", "p"])
    print(w.round(4).to_string())
