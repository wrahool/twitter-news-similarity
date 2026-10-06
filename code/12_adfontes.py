"""
12_adfontes.py -- re-run the analysis with the Ad Fontes Media content-based
bias score (media_slant2.csv) in place of the Pew audience slant.

Ad Fontes scores are centred at 0 (negative = left, positive = right), so blocs
use the sign: right bloc = WSJ, NY Post, Washington Examiner, Fox News, Daily
Caller, Breitbart (6 outlets). Four Ad Fontes entries not in the Twitter
sample are dropped (PBS, Vox, Rush Limbaugh, Sean Hannity).

Runs, on the averaged scores and on each of the five samples:
  every headline estimate in 06_multisample.estimate() (delta, beta, H3,
  asymmetry, isolation, bloc segregation, year-specific gradients, drop-gap and
  rank-distance robustness, leave-one-out);
plus cross-sectional checks on pair means (slope on D; with outlet FE; the
outlet-type regression), a Pew vs Ad Fontes comparison in standardised units
(per SD of pair distance), and a heatmap ordered by the Ad Fontes score.

Outputs: tables/adfontes_long.csv, tables/adfontes_summary.csv,
         tables/adfontes_loo.csv, tables/adfontes_cross_section.csv,
         tables/adfontes_type_regression.csv, tables/adfontes_vs_pew.csv,
         figures/figS10_adfontes_vs_pew.(pdf|png),
         figures/fig_heatmap_political_adfontes.(pdf|png)
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt

ms = importlib.import_module("06_multisample")
ot = importlib.import_module("08_outlet_type")
tr = importlib.import_module("10_type_regression")
hm = importlib.import_module("11_heatmap")
fl = importlib.import_module("07_first_last")
set_style()
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")


def run_all_samples(source):
    common.SLANT_SOURCE = source
    long, loos = [], []
    for lab, path in [(str(k), sample_path(k)) for k in ms.SAMPLES] + [("avg", AVG)]:
        common.DATA_SIM = path
        r, l = ms.estimate(lab)
        long += r; loos += l
    long = pd.DataFrame(long); loos = pd.DataFrame(loos)
    return long, loos, ms.combine(long)


def cross_section(source):
    common.SLANT_SOURCE = source
    common.DATA_SIM = AVG
    p, slant = build_panel()
    pm = p.groupby(["domain", "pair", "o1", "o2", "D"])[MEASURES].mean().reset_index()
    right = [o for o, v in slant.items() if v > BLOC_CUT[source]]
    rows = []
    for d in DOMAINS:
        for m in ["tfidf", "sbert"]:
            x = pm[pm.domain == d].reset_index(drop=True)
            x["t1"], x["t2"] = x.o1.map(ot.TYPE), x.o2.map(ot.TYPE)
            for kind, lab in [("none", "D only"), ("outlet", "D + outlet FE")]:
                r = ot.xsec(x, m, kind)
                rows.append(dict(source=source, domain=d, measure=m, model=lab, slope=r["slope"], se=r["se"],
                                 p=r["p"], r2=r["r2"], sd_D=x.D.std()))
            y = x[~x.o1.isin(right) & ~x.o2.isin(right)].reset_index(drop=True)
            b, se, pv, rr = fl.ols_dyadic(y, m)
            rows.append(dict(source=source, domain=d, measure=m, model="D only, left-of-centre outlets",
                             slope=b[1], se=se[1], p=pv[1], r2=rr ** 2, sd_D=y.D.std()))
    typ = []
    for d in DOMAINS:
        for m in ["tfidf", "sbert"]:
            r = tr.fit(pm[pm.domain == d].reset_index(drop=True), m)
            r.insert(0, "measure", m); r.insert(0, "domain", d); r.insert(0, "source", source)
            typ.append(r)
    return pd.DataFrame(rows), pd.concat(typ), p, slant


COMPARE = [
    ("main_beta", "nonpol", "tfidf", "β non-political, TF-IDF (H1)"),
    ("main_beta", "nonpol", "sbert", "β non-political, SBERT (H1)"),
    ("main_delta", "pol", "tfidf", "δ political, TF-IDF (H2)"),
    ("main_delta", "pol", "sbert", "δ political, SBERT (H2)"),
    ("main_delta", "pol", "gap", "δ political, G (framing)"),
    ("main_delta", "nonpol", "tfidf", "δ non-political, TF-IDF"),
    ("main_delta", "nonpol", "sbert", "δ non-political, SBERT"),
    ("h3_diff", "pol-nonpol", "tfidf", "H3 difference, TF-IDF"),
    ("h3_diff", "pol-nonpol", "sbert", "H3 difference, SBERT"),
    ("asym_M", "pol", "sbert", "M×t political, SBERT (RQ)"),
    ("asym_M", "nonpol", "sbert", "M×t non-political, SBERT (RQ)"),
    ("iso_lambda", "pol", "sbert", "Isolation λ political, SBERT (RQ)"),
    ("iso_lambda", "pol", "tfidf", "Isolation λ political, TF-IDF (RQ)"),
    ("bloc_RR", "pol", "sbert", "Right–right trend political, SBERT"),
    ("bloc_RR", "pol", "tfidf", "Right–right trend political, TF-IDF"),
    ("bloc_phi", "pol", "tfidf", "Bloc segregation φ political, TF-IDF"),
    ("bloc_phi", "pol", "sbert", "Bloc segregation φ political, SBERT"),
    ("rob_dropgap_delta", "pol", "tfidf", "δ political TF-IDF, excl. Breitbart & Fox"),
]


def plot_compare(pew, af, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    tcrit = stats.t.ppf(.975, 24)
    ax.axvspan(-tcrit, tcrit, color="#eeeeee", zorder=0)
    ax.axvline(0, color="#555555", lw=0.8)
    for y, (fam, dom, mea, lab) in enumerate(COMPARE[::-1]):
        for df, col, mk, off, name in [(pew, OI["grey"], "o", 0.15, "Pew audience slant"),
                                       (af, OI["vermillion"], "D", -0.15, "Ad Fontes content score")]:
            g = df[(df.family == fam) & (df.domain == dom) & (df.measure == mea)]
            ind = g[g["sample"] != "avg"]; a = g[g["sample"] == "avg"].iloc[0]
            ax.scatter(ind.est / ind.se, [y + off] * len(ind), s=10, color=col, alpha=0.45, zorder=2)
            ax.scatter([a.est / a.se], [y + off], s=40, marker=mk, color=col, zorder=3,
                       label=name if y == 0 else None)
    ax.set_yticks(range(len(COMPARE)))
    ax.set_yticklabels([c[3] for c in COMPARE[::-1]], fontsize=7.5)
    ax.set_xlabel("t statistic (large marker: averaged scores; small: samples 1–5)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    orig_sim, orig_src = common.DATA_SIM, common.SLANT_SOURCE
    af_long, af_loo, af_sum = run_all_samples("adfontes")
    af_long.to_csv(os.path.join(TAB, "adfontes_long.csv"), index=False)
    af_loo.to_csv(os.path.join(TAB, "adfontes_loo.csv"), index=False)
    af_sum.to_csv(os.path.join(TAB, "adfontes_summary.csv"), index=False)
    pew_long = pd.read_csv(os.path.join(TAB, "multisample_long.csv"), dtype={"sample": str})

    xs_af, typ_af, p_af, slant_af = cross_section("adfontes")
    xs_pew, typ_pew, _, _ = cross_section("pew")
    xs = pd.concat([xs_pew, xs_af])
    xs["slope_per_sd"] = xs.slope * xs.sd_D
    xs.to_csv(os.path.join(TAB, "adfontes_cross_section.csv"), index=False)
    pd.concat([typ_pew, typ_af]).to_csv(os.path.join(TAB, "adfontes_type_regression.csv"), index=False)

    # standardised comparison of distance-based terms (per SD of pair distance)
    sd = {src: xs[(xs.source == src) & (xs.model == "D only")].sd_D.iloc[0] for src in ["pew", "adfontes"]}
    rows = []
    for fam, dom, mea, lab in COMPARE:
        for src, df in [("pew", pew_long), ("adfontes", af_long)]:
            a = df[(df.family == fam) & (df.domain == dom) & (df.measure == mea) & (df["sample"] == "avg")].iloc[0]
            k = df[(df.family == fam) & (df.domain == dom) & (df.measure == mea) & (df["sample"] != "avg")]
            scale = sd[src] if fam in ("main_delta", "h3_diff", "rob_dropgap_delta") else 1.0
            rows.append(dict(estimate=lab, source=src, est=a.est, se=a.se, p=a.p, t=a.est / a.se,
                             est_per_sd_D=a.est * scale, n_sig_samples=int((k.p < .05).sum())))
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(os.path.join(TAB, "adfontes_vs_pew.csv"), index=False)
    plot_compare(pew_long, af_long, os.path.join(FIG, "figS10_adfontes_vs_pew"))

    # heatmap ordered by Ad Fontes
    common.SLANT_SOURCE = "adfontes"
    order = sorted(slant_af, key=lambda o: (slant_af[o], o))
    pm = p_af.groupby(["domain", "o1", "o2"])[["tfidf", "sbert"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, m in zip(axes, ["tfidf", "sbert"]):
        M = hm.matrix(pm[pm.domain == "pol"], m, order)
        # draw() marks the bloc boundary at the first outlet above the Pew midpoint; patch for sign rule
        _orig = {k: v for k, v in slant_af.items()}
        shifted = {k: v + 3.0 for k, v in _orig.items()}     # so that "> 3" == "> 0"
        hm.draw(ax, M, order, shifted, f"Political content: {MEASURE_LABEL[m]} (ordered by Ad Fontes)",
                "Mean similarity (line = median pair)", fig)
        ax.set_xticklabels([f"{hm.NAMES[o]} ({slant_af[o]:+.1f})" for o in order], rotation=90, fontsize=6.3)
        ax.set_yticklabels([f"{hm.NAMES[o]} ({slant_af[o]:+.1f})" for o in order], fontsize=6.3)
    fig.tight_layout()
    path = os.path.join(FIG, "fig_heatmap_political_adfontes")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)
    save_fallback_log("adfontes")
    common.DATA_SIM, common.SLANT_SOURCE = orig_sim, orig_src

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
    print(cmp_.round(4).to_string(index=False))
    print(xs.round(4).to_string(index=False))
    cols = ["family", "domain", "measure", "term", "mean_est", "p_rubin", "n_sig05_same_sign", "avg_est", "avg_p"]
    print(af_sum[af_sum.family != "year_theta"][cols].round(4).to_string())
    print(af_sum[(af_sum.family == "year_theta") & (af_sum.avg_p < .1)][cols].round(4).to_string())
    print(af_loo[af_loo["sample"] == "avg"].round(4).to_string())
    print(typ_af[typ_af.domain == "pol"].round(4).to_string())
