"""
06_multisample.py -- how robust are the findings across the five independent
samples of 500 posts per outlet-month?

For each of samples 1-5, and for "avg" (the pair-month mean of the five
samples' scores, which reduces post-sampling noise), this re-estimates every
headline quantity on the primary window (2015-11 to 2023-01):

  main_delta   delta (D x t), Model A                     9 domain x measure cells
  main_beta    beta (common linear trend), Model B        9
  h3_diff      D x t x Political, stacked model           3
  asym_M       M x t (pair midpoint), Model A + M x t     9
  iso_lambda   slant x t, outlet-month isolation model    9
  bloc_phi     Between x t, composition-robust segreg.    9
  bloc_RR      right-right vs left-left trend             9
  year_theta   D x year (ref. 2016)                       9 x 6
  rob_dropgap_delta / rob_rank_delta / rob_dropgap_h3 / rob_rank_h3
  loo_*        leave-one-outlet-out summaries for political delta

Combining across samples 1-5 (Rubin's rules, m = 5):
  mean estimate, between-sample SD (B), mean within-sample dyadic SE (W),
  total SE = sqrt(W^2_mean + (1 + 1/m) B^2); p from t with 24 df.
The total SE is conservative: the within-sample SE already reflects some of
the post-sampling noise through the residuals.

Outputs: tables/multisample_long.csv, tables/multisample_summary.csv,
         tables/multisample_loo.csv, tables/multisample_sample_checks.csv,
         figures/figS5_multisample_tstats.(pdf|png)
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt
from scipy import stats

mm = importlib.import_module("02_main_models")
ext = importlib.import_module("03_extensions")
rob = importlib.import_module("04_robustness")
set_style()

WORK = os.path.join(OUT, "multisample")
os.makedirs(WORK, exist_ok=True)
SAMPLES = [1, 2, 3, 4, 5]
GAP_OUTLETS = ["BreitbartNews", "FoxNews"]
MCOLS = ["TF-IDF-All", "SBERT-All", "TF-IDF-Pol", "SBERT-Pol", "TF-IDF-NonPol", "SBERT-NonPol"]


def make_avg_file():
    frames = [pd.read_csv(sample_path(k)) for k in SAMPLES]
    key = ["Month", "Outlet 1", "Outlet 2"]
    a = pd.concat(frames).groupby(key, as_index=False)[MCOLS].mean()
    a.insert(1, "Sample", 0)
    path = os.path.join(WORK, "sample_avg_monthly_pairwise_similarity.csv")
    a.to_csv(path, index=False)
    return path


def sample_checks():
    """Cross-sample agreement of the raw scores (primary window)."""
    frames = {k: pd.read_csv(sample_path(k)) for k in SAMPLES}
    key = ["Month", "Outlet 1", "Outlet 2"]
    base = frames[1].set_index(key)
    rows = []
    for k in SAMPLES[1:]:
        o = frames[k].set_index(key).loc[base.index]
        assert len(o) == len(base)
        for c in MCOLS:
            rows.append(dict(sample=k, measure=c, corr_with_s1=np.corrcoef(o[c], base[c])[0, 1],
                             identical_share=np.isclose(o[c], base[c], atol=1e-9).mean()))
    allf = pd.concat(frames.values())
    allf = allf[allf.Month >= PRIMARY_START]
    b = allf.groupby(key)[MCOLS].std().mean()
    w = allf[allf.Sample == 1].groupby("Month")[MCOLS].std().mean()
    disp = pd.DataFrame({"between_sample_sd": b, "cross_pair_sd_within_month": w, "ratio": b / w})
    return pd.DataFrame(rows), disp


def rec(rows, sample, family, domain, measure, term, est, se, p):
    rows.append(dict(sample=sample, family=family, domain=domain, measure=measure, term=term,
                     est=est, se=se, p=p))


def estimate(sample_label):
    rows, loo_rows = [], []
    p, slant = build_panel()

    r = mm.run_main(p)
    for _, x in r.iterrows():
        rec(rows, sample_label, "main_delta", x.domain, x.measure, "D x t", x.delta, x.se, x.p)
        rec(rows, sample_label, "main_beta", x.domain, x.measure, "t", x.beta, x.beta_se, x.beta_p)
    h = mm.run_h3(p, False)
    for _, x in h.iterrows():
        rec(rows, sample_label, "h3_diff", "pol-nonpol", x.measure, "D x t x Pol", x["diff"], x.diff_se, x.diff_p)

    q = ext.add_terms(p)
    for _, x in ext.asym_pair(q).iterrows():
        rec(rows, sample_label, "asym_M", x.domain, x.measure, "M x t", x.mu, x.mu_se, x.mu_p)
    lam, _ = ext.isolation_models(ext.outlet_panel(q, slant))
    for _, x in lam.iterrows():
        rec(rows, sample_label, "iso_lambda", x.domain, x.measure, "slant x t", x.lam, x.se, x.p)
    tr = ext.bloc_trends(ext.bloc_series(q), q)
    for _, x in tr.iterrows():
        rec(rows, sample_label, "bloc_phi", x.domain, x.measure, "Between x t", x.phi_between_t, x.phi_se, x.phi_p)
        rec(rows, sample_label, "bloc_RR", x.domain, x.measure, "RR x t", x.RR_vs_LL, x.RR_se, x.RR_p)
    for _, x in ext.year_gradient(q).iterrows():
        if x.year != 2015:
            rec(rows, sample_label, "year_theta", x.domain, x.measure, f"D x {int(x.year)}", x.coef, x.se, x.p)

    # robustness: drop Breitbart & Fox; rank distance
    raw = load_raw()[0].rename(columns={"Month": "month", "Outlet 1": "o1", "Outlet 2": "o2"})
    raw = raw[(raw.month >= PRIMARY_START) & ~(raw.o1.isin(GAP_OUTLETS) | raw.o2.isin(GAP_OUTLETS))]
    pdg, _ = panel_from_frame(raw)
    for _, x in mm.run_main(pdg).iterrows():
        rec(rows, sample_label, "rob_dropgap_delta", x.domain, x.measure, "D x t", x.delta, x.se, x.p)
    for _, x in mm.run_h3(pdg, False).iterrows():
        rec(rows, sample_label, "rob_dropgap_h3", "pol-nonpol", x.measure, "D x t x Pol", x["diff"], x.diff_se, x.diff_p)
    for _, x in mm.run_main(p, dvar="R").iterrows():
        rec(rows, sample_label, "rob_rank_delta", x.domain, x.measure, "R x t", x.delta, x.se, x.p)
    pr = p.copy(); pr["D"] = pr["R"]
    for _, x in mm.run_h3(pr, False).iterrows():
        rec(rows, sample_label, "rob_rank_h3", "pol-nonpol", x.measure, "R x t x Pol", x["diff"], x.diff_se, x.diff_p)

    # leave-one-outlet-out, political delta
    loo = rob.leave_one_out(p, slant)
    for m, g in loo.groupby("measure"):
        o = g[g.dropped != "(none)"]
        mn, mx = o.loc[o.delta.idxmin()], o.loc[o.delta.idxmax()]
        loo_rows.append(dict(sample=sample_label, measure=m, full=g[g.dropped == "(none)"].delta.iloc[0],
                             min=mn.delta, min_dropped=mn.dropped, max=mx.delta, max_dropped=mx.dropped,
                             n_pos=int((o.delta > 0).sum()), n_sig=int((o.p < .05).sum()),
                             drop_breitbart=o[o.dropped == "BreitbartNews"].delta.iloc[0]))
    return rows, loo_rows


def combine(long):
    ind = long[long["sample"].isin([str(k) for k in SAMPLES])]
    keys = ["family", "domain", "measure", "term"]
    out = []
    for k, g in ind.groupby(keys, sort=False):
        m = len(g)
        qbar = g.est.mean(); B = g.est.var(ddof=1); W = (g.se ** 2).mean()
        T = np.sqrt(W + (1 + 1 / m) * B)
        sign = np.sign(qbar)
        av = long[(long["sample"] == "avg") & (long.family == k[0]) & (long.domain == k[1]) &
                  (long.measure == k[2]) & (long.term == k[3])].iloc[0]
        s1 = g[g["sample"] == "1"].iloc[0]
        out.append(dict(zip(keys, k), s1_est=s1.est, s1_p=s1.p,
                        mean_est=qbar, min_est=g.est.min(), max_est=g.est.max(),
                        between_sd=np.sqrt(B), mean_within_se=np.sqrt(W),
                        between_over_within=np.sqrt(B / W), total_se=T,
                        p_rubin=2 * stats.t.sf(abs(qbar / T), 24),
                        n_same_sign=int((np.sign(g.est) == sign).sum()),
                        n_sig05=int((g.p < .05).sum()),
                        n_sig05_same_sign=int(((g.p < .05) & (np.sign(g.est) == sign)).sum()),
                        avg_est=av.est, avg_se=av.se, avg_p=av.p))
    return pd.DataFrame(out)


HEADLINE = [
    ("main_beta", "nonpol", "tfidf", "β non-political, TF-IDF (H1)"),
    ("main_beta", "nonpol", "sbert", "β non-political, SBERT (H1)"),
    ("main_beta", "pol", "tfidf", "β political, TF-IDF"),
    ("main_beta", "pol", "sbert", "β political, SBERT"),
    ("main_delta", "pol", "tfidf", "δ political, TF-IDF (H2)"),
    ("main_delta", "pol", "sbert", "δ political, SBERT (H2)"),
    ("main_delta", "pol", "gap", "δ political, G (framing)"),
    ("main_delta", "nonpol", "tfidf", "δ non-political, TF-IDF"),
    ("main_delta", "nonpol", "sbert", "δ non-political, SBERT"),
    ("main_delta", "all", "sbert", "δ all posts, SBERT"),
    ("h3_diff", "pol-nonpol", "tfidf", "H3 difference, TF-IDF"),
    ("h3_diff", "pol-nonpol", "sbert", "H3 difference, SBERT"),
    ("asym_M", "pol", "sbert", "M×t political, SBERT (RQ)"),
    ("asym_M", "nonpol", "sbert", "M×t non-political, SBERT (RQ)"),
    ("iso_lambda", "pol", "sbert", "Isolation λ political, SBERT (RQ)"),
    ("iso_lambda", "all", "sbert", "Isolation λ all, SBERT (RQ)"),
    ("bloc_RR", "pol", "sbert", "Right–right trend political, SBERT"),
    ("bloc_RR", "nonpol", "gap", "Right–right trend non-political, G"),
    ("bloc_phi", "pol", "tfidf", "Bloc segregation φ political, TF-IDF"),
    ("bloc_phi", "nonpol", "tfidf", "Bloc segregation φ non-political, TF-IDF"),
]


def plot_t(long, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    tcrit = stats.t.ppf(.975, 24)
    ax.axvspan(-tcrit, tcrit, color="#eeeeee", zorder=0)
    ax.axvline(0, color="#555555", lw=0.8)
    for y, (fam, dom, mea, lab) in enumerate(HEADLINE[::-1]):
        g = long[(long.family == fam) & (long.domain == dom) & (long.measure == mea)]
        ind = g[g["sample"] != "avg"]
        ax.scatter(ind.est / ind.se, [y] * len(ind), s=16, color=OI["blue"], alpha=0.75, zorder=3,
                   label="Samples 1–5" if y == 0 else None)
        a = g[g["sample"] == "avg"].iloc[0]
        ax.scatter([a.est / a.se], [y], s=46, marker="D", facecolor="white", edgecolor=OI["vermillion"],
                   lw=1.4, zorder=4, label="Averaged scores" if y == 0 else None)
    ax.set_yticks(range(len(HEADLINE)))
    ax.set_yticklabels([h[3] for h in HEADLINE[::-1]], fontsize=7.5)
    ax.set_xlabel("t statistic (estimate / dyadic SE; outlet-clustered SE for λ)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    suptitle(fig, "Headline estimates across the five samples (shaded: |t| < 2.06, p > .05)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    checks, disp = sample_checks()
    checks.to_csv(os.path.join(TAB, "multisample_sample_checks.csv"), index=False)
    disp.to_csv(os.path.join(TAB, "multisample_dispersion.csv"))
    avg_path = make_avg_file()
    original = common.DATA_SIM
    long, loos = [], []
    for lab, path in [(str(k), sample_path(k)) for k in SAMPLES] + [("avg", avg_path)]:
        common.DATA_SIM = path
        r, l = estimate(lab)
        long += r; loos += l
        print("done sample", lab, flush=True)
    common.DATA_SIM = original
    long = pd.DataFrame(long); loos = pd.DataFrame(loos)
    long.to_csv(os.path.join(TAB, "multisample_long.csv"), index=False)
    loos.to_csv(os.path.join(TAB, "multisample_loo.csv"), index=False)
    summ = combine(long)
    summ.to_csv(os.path.join(TAB, "multisample_summary.csv"), index=False)
    plot_t(long, os.path.join(FIG, "figS5_multisample_tstats"))
    save_fallback_log("multisample")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
    print(disp.round(4)); print(checks.groupby("measure")[["corr_with_s1", "identical_share"]].mean().round(3))
    cols = ["family", "domain", "measure", "term", "s1_est", "s1_p", "mean_est", "min_est", "max_est",
            "between_over_within", "p_rubin", "n_same_sign", "n_sig05_same_sign", "avg_est", "avg_p"]
    print(summ[summ.family != "year_theta"][cols].round(4).to_string())
    print(summ[summ.family == "year_theta"][cols].round(4).to_string())
    print(loos.round(4).to_string())
