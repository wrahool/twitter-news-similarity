"""
07_first_last.py -- descriptive cross-sections: pairwise similarity as a
function of ideological distance D in the first and last month.

Months compared: November 2015 (first month of the primary window) and
January 2023 (last month); November 2014 (first month in the data, before
the TF-IDF break) is reported for reference. All 300 pairs are present in
each of these months, so the cross-sections are not affected by gaps.

Scores are the pair-month average of the five samples (less sampling noise);
sample 1 alone is reported as a check.

For each domain x measure x month: OLS of similarity on D (300 pairs) with
dyadic cluster-robust SEs (pairs sharing an outlet may be correlated).
Change in gradient: OLS of (Sim_last - Sim_first) on D, which is the
two-period version of Model A's delta x (t_last - t_first).

Outputs: tables/first_last_gradients.csv, tables/first_last_change.csv,
         figures/figS6_first_last_gradient.(pdf|png)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt

set_style()
FIRST, LAST, PREBREAK = "2014-11", "2023-01", "2015-11"   # v2: PREBREAK slot now holds Nov 2015 for reference
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")


def ols_dyadic(df, y, x="D"):
    """Cross-sectional OLS y = a + b x with dyadic SEs (one row per pair)."""
    X = np.column_stack([np.ones(len(df)), df[x].values])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ df[y].values
    e = df[y].values - X @ b
    V, G = vcov_dyadic(X, e, df.o1.values, df.o2.values, df.pair.values, XtX_inv)
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(b / se), G - 1)
    r = np.corrcoef(df[x], df[y])[0, 1]
    return b, se, p, r


def run(label, path):
    common.DATA_SIM = path
    p, _ = build_panel(start=None)          # G standardised over full window
    rows, chg = [], []
    for dom in DOMAINS:
        for m in MEASURES:
            x = p[p.domain == dom]
            for mth in [PREBREAK, FIRST, LAST]:
                c = x[x.month == mth]
                b, se, pv, r = ols_dyadic(c, m)
                rows.append(dict(data=label, domain=dom, measure=m, month=mth, n_pairs=len(c),
                                 mean_sim=c[m].mean(), intercept=b[0], slope=b[1], slope_se=se[1], slope_p=pv[1],
                                 r=r, pred_D0=b[0], pred_Dmax=b[0] + b[1] * c.D.max()))
            w = x[x.month.isin([FIRST, LAST])].pivot_table(index=["pair", "o1", "o2", "D"], columns="month",
                                                               values=m).dropna().reset_index()
            w["chg"] = w[LAST] - w[FIRST]
            b, se, pv, r = ols_dyadic(w, "chg")
            chg.append(dict(data=label, domain=dom, measure=m, n_pairs=len(w), mean_change=w.chg.mean(),
                            slope_change=b[1], se=se[1], p=pv[1]))
    return pd.DataFrame(rows), pd.DataFrame(chg), p


def plot(p, g, path):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), sharex=True)
    for i, dom in enumerate(DOMAINS):
        for j, m in enumerate(["tfidf", "sbert"]):
            ax = axes[i, j]
            x = p[p.domain == dom]
            for mth, col, lab, mk in [(FIRST, OI["grey"], "Nov 2014", "o"), (LAST, MEASURE_COL[m], "Jan 2023", "o")]:
                c = x[x.month == mth]
                ax.scatter(c.D, c[m], s=9, alpha=0.55, color=col, edgecolor="none", label=lab)
                r = g[(g.domain == dom) & (g.measure == m) & (g.month == mth)].iloc[0]
                xs = np.linspace(0, c.D.max(), 10)
                ax.plot(xs, r.intercept + r.slope * xs, color="#333333" if mth == FIRST else col, lw=1.6,
                        ls="--" if mth == FIRST else "-")
            f = g[(g.domain == dom) & (g.measure == m) & (g.month == FIRST)].iloc[0]
            l = g[(g.domain == dom) & (g.measure == m) & (g.month == LAST)].iloc[0]
            ax.text(0.98, 0.03, f"slope Nov 2014: {f.slope:+.3f} ({f.slope_se:.3f})\n"
                                f"slope Jan 2023: {l.slope:+.3f} ({l.slope_se:.3f})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
            if i == 0:
                ax.set_title(MEASURE_LABEL[m])
            if j == 0:
                ax.set_ylabel(f"{DOMAIN_LABEL[dom]}\nPairwise similarity")
            if i == 2:
                ax.set_xlabel("Ideological distance $D$ (|slant$_i$ − slant$_j$|)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), markerscale=2)
    suptitle(fig, "Similarity vs ideological distance, first and last month", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    original = common.DATA_SIM
    g_avg, c_avg, p_avg = run("avg of 5 samples", AVG)
    g_s1, c_s1, _ = run("sample 1", sample_path(1))
    common.DATA_SIM = original
    pd.concat([g_avg, g_s1]).to_csv(os.path.join(TAB, "first_last_gradients.csv"), index=False)
    pd.concat([c_avg, c_s1]).to_csv(os.path.join(TAB, "first_last_change.csv"), index=False)
    plot(p_avg, g_avg, os.path.join(FIG, "figS6_first_last_gradient"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(g_avg[["domain", "measure", "month", "mean_sim", "slope", "slope_se", "slope_p", "r", "pred_D0", "pred_Dmax"]].round(4).to_string())
    print(pd.concat([c_avg, c_s1]).round(4).to_string())
