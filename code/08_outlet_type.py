"""
08_outlet_type.py -- does ideological distance predict coverage similarity once
we account for what kind of outlet each one is?

Outlet types (EDIT HERE if you disagree; these are judgment calls):
  broadcast   ABC, CBS, NBC, CNN, Fox News, MSNBC, NPR       (TV and public radio)
  newspaper   NYT, WaPo, WSJ, USA Today, NY Post              (U.S. daily papers)
  magazine    TIME, Newsweek, Washington Examiner             (weekly/news magazines)
  digital     HuffPost, BuzzFeed News, Business Insider, VICE, Politico,
              The Hill, Daily Caller, Breitbart                (digital-first)
  uk          The Guardian, BBC World                         (U.K.-based)

Part 1 -- cross-section of pair means over the primary window (300 pairs):
  (a) Sim ~ D
  (b) Sim ~ D + same-type indicator
  (c) Sim ~ D + type-pair fixed effects (one per unordered combination of types)
  (d) Sim ~ D + outlet fixed effects (a_i + a_j: each outlet's general
      distinctiveness, e.g. the Guardian being dissimilar to everyone)
  plus R^2 of each set of controls without D.
Part 2 -- panel (Model A) delta with additional trend controls:
  Model A; + type-pair x t trends; + outlet-specific x t trends.
All SEs dyadic cluster-robust. Scores: average of the five samples.

Outputs: tables/outlet_type_cross_section.csv, tables/outlet_type_panel.csv,
         tables/outlet_types.csv
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *

TYPE = {
    "ABC": "broadcast", "CBSNews": "broadcast", "NBCNews": "broadcast", "CNN": "broadcast",
    "FoxNews": "broadcast", "MSNBC": "broadcast", "NPR": "broadcast",
    "nytimes": "newspaper", "washingtonpost": "newspaper", "WSJ": "newspaper", "USATODAY": "newspaper",
    "nypost": "newspaper",
    "TIME": "magazine", "Newsweek": "magazine", "dcexaminer": "magazine",
    "HuffPost": "digital", "BuzzFeedNews": "digital", "BusinessInsider": "digital", "VICE": "digital",
    "politico": "digital", "thehill": "digital", "DailyCaller": "digital", "BreitbartNews": "digital",
    "guardian": "uk", "BBCWorld": "uk",
}
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")


def dummies(df, kind):
    if kind == "same":
        return (df.t1 == df.t2).astype(float).values[:, None], ["same_type"]
    if kind == "typepair":
        tp = df.apply(lambda r: "|".join(sorted([r.t1, r.t2])), axis=1)
        d = pd.get_dummies(tp, drop_first=True).astype(float)
        return d.values, list(d.columns)
    if kind == "outlet":
        outs = sorted(set(df.o1) | set(df.o2))
        M = np.zeros((len(df), len(outs)))
        for k, o in enumerate(outs):
            M[:, k] = ((df.o1 == o) | (df.o2 == o)).astype(float)
        return M[:, 1:], outs[1:]          # drop one (collinear with intercept)
    return np.zeros((len(df), 0)), []


def xsec(df, y, kind, with_D=True):
    """OLS; returns D coefficient with dyadic SE (via FWL) and R^2."""
    C, _ = dummies(df, kind)
    Z = np.column_stack([np.ones(len(df)), C])
    X = np.column_stack([Z, df.D.values]) if with_D else Z
    b, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    e = df[y].values - X @ b
    r2 = 1 - (e @ e) / ((df[y] - df[y].mean()) ** 2).sum()
    if not with_D:
        return dict(r2=r2)
    # FWL: residualise D on the controls, then dyadic SE on that single regressor
    gz, *_ = np.linalg.lstsq(Z, df.D.values, rcond=None)
    Dt = (df.D.values - Z @ gz)[:, None]
    XtX_inv = np.linalg.inv(Dt.T @ Dt)
    V, G = vcov_dyadic(Dt, e, df.o1.values, df.o2.values, df.pair.values, XtX_inv)
    se = float(np.sqrt(V[0, 0]))
    return dict(slope=b[-1], se=se, p=2 * stats.t.sf(abs(b[-1] / se), G - 1), r2=r2)


if __name__ == "__main__":
    common.DATA_SIM = AVG
    p, slant = build_panel()
    p["t1"], p["t2"] = p.o1.map(TYPE), p.o2.map(TYPE)
    assert p.t1.notna().all() and p.t2.notna().all()
    pd.DataFrame({"outlet": list(TYPE), "type": list(TYPE.values()),
                  "slant": [slant[o] for o in TYPE]}).sort_values(["type", "slant"]).to_csv(
        os.path.join(TAB, "outlet_types.csv"), index=False)

    pm = p.groupby(["domain", "pair", "o1", "o2", "t1", "t2", "D"])[MEASURES].mean().reset_index()
    rows = []
    for dom in DOMAINS:
        x = pm[pm.domain == dom].reset_index(drop=True)
        for m in ["tfidf", "sbert"]:
            for kind, lab in [("none", "(a) D only"), ("same", "(b) + same type"),
                              ("typepair", "(c) + type-pair FE"), ("outlet", "(d) + outlet FE")]:
                r = xsec(x, m, kind)
                r0 = xsec(x, m, kind, with_D=False) if kind != "none" else dict(r2=0.0)
                rows.append(dict(domain=dom, measure=m, model=lab, slope_D=r["slope"], se=r["se"], p=r["p"],
                                 r2_full=r["r2"], r2_controls_only=r0["r2"], r2_added_by_D=r["r2"] - r0["r2"]))
    xs = pd.DataFrame(rows)
    xs.to_csv(os.path.join(TAB, "outlet_type_cross_section.csv"), index=False)

    # Panel: Model A with extra trends
    pmean = p.groupby("pair").D.first().mean()
    p["Dt"] = (p.D - pmean) * p.t
    tp = p.apply(lambda r: "|".join(sorted([r.t1, r.t2])), axis=1)
    tps = sorted(tp.unique())
    for k in tps[1:]:
        p[f"tp_{k}"] = (tp == k).astype(float) * p.t
    outs = sorted(slant)
    for o in outs[1:]:
        p[f"ot_{o}"] = ((p.o1 == o) | (p.o2 == o)).astype(float) * p.t
    tp_cols = [f"tp_{k}" for k in tps[1:]]; ot_cols = [f"ot_{o}" for o in outs[1:]]
    prow = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            for lab, extra in [("Model A", []), ("+ type-pair x t", tp_cols), ("+ outlet x t", ot_cols)]:
                r = fe_ols(x, m, ["Dt"] + extra, ["pair", "month"])
                t = r["tab"].loc["Dt"]
                prow.append(dict(domain=dom, measure=m, model=lab, delta=t.coef, se=t.se_dyadic, p=t.p_dyadic,
                                 vcov_fix=r.get("dyadic_fallback")))
    pn = pd.DataFrame(prow)
    pn.to_csv(os.path.join(TAB, "outlet_type_panel.csv"), index=False)
    save_fallback_log("outlet_type")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(xs.round(4).to_string())
    print(pn.round(4).to_string())
