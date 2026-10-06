"""
04_robustness.py -- Step 6 robustness checks.

Each alternative sample/specification re-estimates the 9 Model-A deltas, the
Model-B common trends (beta) and the H3 pooled difference:
  primary        2015-11..2023-01, unbalanced panel (reference)
  pre_break_incl full window 2014-11..2023-01 (includes the TF-IDF break)
  to_2022_10     drop months after Oct 2022 (ownership change)
  rank_D         rank distance instead of D
  interpolated   linear interpolation of interior gaps
  drop_gap       drop all pairs involving an outlet with a >= 6-month gap
                 (Breitbart, Fox)
  pre_280 / post_280  split at Nov 2017
Plus: D x Post-Nov-2017 added to Model A; median-split blocs for the bloc
segregation analysis; leave-one-outlet-out for the political models.

Outputs: tables/robustness_delta.csv, tables/table2_robustness.md,
         tables/robust_280_interaction.csv, tables/loo_political.csv,
         tables/bloc_segregation_trends_median.csv,
         figures/fig5_leave_one_out.(pdf|png)
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib.pyplot as plt

mm = importlib.import_module("02_main_models")
ext = importlib.import_module("03_extensions")
set_style()

GAP_OUTLETS = ["BreitbartNews", "FoxNews"]   # outlets with a gap >= 6 months


def samples():
    out = {}
    out["primary"], _ = build_panel()
    out["from_2015_11"], _ = build_panel(start=ALT_START)
    out["to_2022_10"], _ = build_panel(end=MUSK_LAST)
    out["interpolated"], _ = panel_from_frame(interpolate_panel())
    out["drop_gap"], _ = panel_from_frame(
        load_raw()[0].rename(columns={"Month": "month", "Outlet 1": "o1", "Outlet 2": "o2"})
        .query("month >= @PRIMARY_START")
        .pipe(lambda d: d[~(d.o1.isin(GAP_OUTLETS) | d.o2.isin(GAP_OUTLETS))]))
    out["pre_280"], _ = build_panel(end="2017-10")
    out["post_280"], _ = build_panel(start=BREAK_280)
    return out


def robustness_table(allres, h3res, path_md):
    order = ["primary", "from_2015_11", "to_2022_10", "rank_D", "interpolated", "drop_gap", "pre_280", "post_280"]
    lab = {"primary": "Primary (2014-11–2023-01)", "from_2015_11": "Starting Nov 2015 (v1 window)",
           "to_2022_10": "Ends Oct 2022", "rank_D": "Rank distance", "interpolated": "Interpolated gaps",
           "drop_gap": "Drop Breitbart & Fox", "pre_280": "Pre-280 (to 2017-10)", "post_280": "Post-280 (2017-11–)"}
    cols = [(d, m) for d in DOMAINS for m in MEASURES]
    md = ["**Table 2. Robustness of the distance-gradient trend δ (Model A) across samples and specifications**", "",
          "| Specification | " + " | ".join(f"{DOMAIN_LABEL[d]}<br>{MEASURE_LABEL[m].split(' (')[0]}" for d, m in cols)
          + " | H3 diff TF-IDF | H3 diff SBERT | N pairs | N months |", "|---" * (len(cols) + 5) + "|"]
    for s in order:
        r = allres[allres["sample"] == s]
        h = h3res[(h3res["sample"] == s) & (~h3res.standardised)]
        cells = []
        for d, m in cols:
            x = r[(r.domain == d) & (r.measure == m)].iloc[0]
            cells.append(f"{x.delta:.4f}{stars(x.p)}<br>({x.se:.4f})")
        for m in ["tfidf", "sbert"]:
            x = h[h.measure == m].iloc[0]
            cells.append(f"{x['diff']:.4f}{stars(x.diff_p)}<br>({x.diff_se:.4f})")
        x0 = r.iloc[0]
        md.append(f"| {lab[s]} | " + " | ".join(cells) + f" | {int(x0.n_pairs)} | {int(x0.n_months)} |")
    md += ["", "Notes: δ = coefficient on D×t with pair and month FE; dyadic SEs in parentheses. "
           "Rank distance is in slant-rank units (1–24), so its δ is not on the same scale. "
           "H3 diff = coefficient on D×t×Political in the stacked political + non-political model "
           "(positive = the gradient *flattens* more for political content). + p<.10, * p<.05, ** p<.01, *** p<.001."]
    open(path_md, "w").write("\n".join(md))


def post280_interaction(p):
    p = p.copy()
    pm = p.groupby("pair").D.first().mean()
    p["Dt"] = (p.D - pm) * p.t
    p["Dpost"] = (p.D - pm) * (p.month >= BREAK_280)
    rows = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            r = fe_ols(x, m, ["Dt", "Dpost"], ["pair", "month"])
            t = r["tab"]
            rows.append(dict(domain=dom, measure=m,
                             delta=t.loc["Dt", "coef"], delta_se=t.loc["Dt", "se_dyadic"], delta_p=t.loc["Dt", "p_dyadic"],
                             D_post=t.loc["Dpost", "coef"], D_post_se=t.loc["Dpost", "se_dyadic"],
                             D_post_p=t.loc["Dpost", "p_dyadic"]))
    return pd.DataFrame(rows)


def leave_one_out(p, slant):
    rows = []
    pol = p[p.domain == "pol"]
    outlets = sorted(slant)
    for o in [None] + outlets:
        x = pol if o is None else pol[(pol.o1 != o) & (pol.o2 != o)].copy()
        pm = x.groupby("pair").D.first().mean()
        x = x.assign(Dt=(x.D - pm) * x.t)
        # G keeps the full-sample standardisation so deltas stay on one scale
        for m in MEASURES:
            r = fe_ols(x, m, ["Dt"], ["pair", "month"])
            t = r["tab"].loc["Dt"]
            rows.append(dict(dropped=o or "(none)", slant=slant.get(o, np.nan), measure=m,
                             delta=t.coef, se=t.se_dyadic, p=t.p_dyadic))
    return pd.DataFrame(rows)


def plot_loo(loo, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
    for j, m in enumerate(MEASURES):
        ax = axes[j]
        g = loo[(loo.measure == m) & (loo.dropped != "(none)")].sort_values("delta")
        full = loo[(loo.measure == m) & (loo.dropped == "(none)")].iloc[0]
        y = np.arange(len(g))
        ax.axvline(0, color="#555555", lw=0.8)
        ax.axvline(full.delta, color=OI["vermillion"], lw=1, ls="--", label="All outlets")
        ax.errorbar(g.delta, y, xerr=1.96 * g.se, fmt="o", ms=3, color=MEASURE_COL[m],
                    ecolor=MEASURE_COL[m], elinewidth=0.7, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(g.dropped, fontsize=5.8)
        ax.set_title(MEASURE_LABEL[m])
        ax.set_xlabel("δ (D × t), political content")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    suptitle(fig, "Leave-one-outlet-out estimates of δ for political content (label = outlet dropped; ±1.96 dyadic SE)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    S = samples()
    res, h3 = [], []
    for name, p in S.items():
        res.append(mm.run_main(p, label=name))
        for std in (False, True):
            h = mm.run_h3(p, std); h["sample"] = name; h3.append(h)
    res.append(mm.run_main(S["primary"], label="rank_D", dvar="R"))
    # H3 with rank distance
    pr = S["primary"].copy(); pr["D"] = pr["R"]
    for std in (False, True):
        h = mm.run_h3(pr, std); h["sample"] = "rank_D"; h3.append(h)
    res, h3 = pd.concat(res), pd.concat(h3)
    res.to_csv(os.path.join(TAB, "robustness_delta.csv"), index=False)
    h3.to_csv(os.path.join(TAB, "robustness_h3.csv"), index=False)
    robustness_table(res, h3, os.path.join(TAB, "table2_robustness.md"))

    pi = post280_interaction(S["primary"])
    pi.to_csv(os.path.join(TAB, "robust_280_interaction.csv"), index=False)

    # bloc segregation under median split
    pmed, slant = build_panel(bloc_rule="median")
    pmed = ext.add_terms(pmed)
    seg = ext.bloc_series(pmed)
    trm = ext.bloc_trends(seg, pmed)
    trm.to_csv(os.path.join(TAB, "bloc_segregation_trends_median.csv"), index=False)
    ext.plot_bloc(seg, trm, os.path.join(FIG, "figS1_bloc_segregation_median"), "\nBlocs: median split (right = slant > 2.62; 12 outlets)")

    loo = leave_one_out(S["primary"], slant)
    loo.to_csv(os.path.join(TAB, "loo_political.csv"), index=False)
    plot_loo(loo, os.path.join(FIG, "fig5_leave_one_out"))

    save_fallback_log("robustness")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.pivot_table(index="sample", columns=["domain", "measure"], values="delta").round(4))
    print(res.pivot_table(index="sample", columns=["domain", "measure"], values="p").round(3))
    print(res.pivot_table(index="sample", columns=["domain", "measure"], values="beta").round(4))
    print(res.pivot_table(index="sample", columns=["domain", "measure"], values="beta_p").round(3))
    print(h3[~h3.standardised][["sample", "measure", "diff", "diff_se", "diff_p"]].round(4))
    print(pi.round(4)); print(trm.round(4))
    print(loo.groupby("measure").delta.describe().round(4))
    print(loo[loo.measure == "tfidf"].sort_values("delta")[["dropped", "delta", "p"]].round(4).to_string())
