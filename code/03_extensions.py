"""
03_extensions.py -- Step 5 extensions.

 1. Year-by-year gradient: Sim = a_ij + g_t + sum_y theta_y (Dc x 1[year=y]).
    v2: reference year = 2015 (Nov-Dec 2014 pooled into it). v1 text follows.
    Reference year = 2016. The brief asked for 2014, but 2014 (and most of 2015)
    falls before the primary window. The two 2015 months (Nov-Dec) are pooled
    into the 2016 reference bin, and the single 2023 month (Jan) into 2022.
    Figure: fig1_year_gradient.(pdf|png)
    Also a descriptive cross-sectional gradient (OLS of Sim on D, by year,
    no pair FE) so the reader can see the *level* that theta_y shifts.
 2. Asymmetry: (a) add Mc x t to Model A. (b) Outlet isolation: each
    outlet's mean similarity to all other outlets present that month;
    iso_it = a_i + g_t + lambda (slant_c x t); SEs clustered by outlet
    (G = 25). Outlet-specific trends (net of month FE) are plotted against slant.
    Figure: fig4_isolation.(pdf|png)
 3. Bloc segregation: monthly mean(within-bloc) - mean(between-bloc).
    Trend by OLS with Newey-West SEs (6 lags). Because Fox/Breitbart gaps change
    the bloc composition, we also estimate the composition-robust panel
    analogue Sim = a_ij + g_t + phi (Between x t) (dyadic SEs), and a
    version splitting pair types LL / LR / RR.
    Figure: fig3_bloc_segregation.(pdf|png)
 4. Common trend: month FE gamma_t from Model A (D centred, so gamma_t is
    the trajectory of an average-distance pair; normalised to 2015-11 = 0).
    Figure: fig2_common_trend.(pdf|png)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

set_style()
SHORT = {"BreitbartNews": "Breitbart", "FoxNews": "Fox", "DailyCaller": "Daily Caller",
         "dcexaminer": "DC Examiner", "nypost": "NY Post"}


def add_terms(p):
    p = p.copy()
    pairs = p.groupby("pair")[["D", "M"]].first()
    p["Dc"] = p.D - pairs.D.mean()
    p["Mc"] = p.M - pairs.M.mean()
    p["Dt"] = p.Dc * p.t
    p["Mt"] = p.Mc * p.t
    p["ybin"] = p.year.clip(2015, 2022)   # Nov-Dec 2014 pooled into 2015; Jan 2023 into 2022
    return p


# ---------------------------------------------------------------- 1. year gradient
def year_gradient(p, ref=2015):
    years = sorted(y for y in p.ybin.unique() if y != ref)
    for y in years:
        p[f"Dy{y}"] = p.Dc * (p.ybin == y)
    xv = [f"Dy{y}" for y in years]
    rows = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            r = fe_ols(x, m, xv, ["pair", "month"])
            t = r["tab"]
            rows.append(dict(domain=dom, measure=m, year=ref, coef=0.0, se=0.0, lo=0.0, hi=0.0, p=np.nan))
            for y, v in zip(years, xv):
                rows.append(dict(domain=dom, measure=m, year=y, coef=t.loc[v, "coef"],
                                 se=t.loc[v, "se_dyadic"], lo=t.loc[v, "lo"], hi=t.loc[v, "hi"],
                                 p=t.loc[v, "p_dyadic"]))
    return pd.DataFrame(rows).sort_values(["domain", "measure", "year"])


def descriptive_gradient(p):
    rows = []
    for dom in DOMAINS:
        for m in MEASURES:
            for y, g in p[p.domain == dom].groupby("ybin"):
                b = np.polyfit(g.D, g[m], 1)[0]
                rows.append(dict(domain=dom, measure=m, year=y, slope_on_D=b))
    return pd.DataFrame(rows)


def plot_year_gradient(yg, path):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True)
    for i, dom in enumerate(DOMAINS):
        for j, m in enumerate(MEASURES):
            ax = axes[i, j]
            g = yg[(yg.domain == dom) & (yg.measure == m)]
            ax.axhline(0, color="#555555", lw=0.8)
            ax.fill_between(g.year, g.lo, g.hi, color=MEASURE_COL[m], alpha=0.18, lw=0)
            ax.plot(g.year, g.coef, "-o", color=MEASURE_COL[m], ms=3.5, lw=1.4)
            ax.plot([2015], [0], "o", mfc="white", mec=MEASURE_COL[m], ms=4.5)
            if i == 0:
                ax.set_title(MEASURE_LABEL[m])
            if j == 0:
                ax.set_ylabel(f"{DOMAIN_LABEL[dom]}\nθ (D × year)")
            ax.set_xticks(range(2015, 2023))
            ax.set_xticklabels(["'15*", "'16", "'17", "'18", "'19", "'20", "'21", "'22†"])
    fig.text(0.5, -0.01, "Year", ha="center", fontsize=9)
    suptitle(fig, "Change in the ideological-distance gradient of similarity relative to 2015\n"
                 "(pair and month FE; 95% CIs from dyadic cluster-robust SEs; θ < 0 = distant pairs diverge)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


# ---------------------------------------------------------------- 2. asymmetry
def asym_pair(p):
    rows = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            r = fe_ols(x, m, ["Dt", "Mt"], ["pair", "month"])
            t = r["tab"]
            rows.append(dict(domain=dom, measure=m,
                             delta=t.loc["Dt", "coef"], delta_se=t.loc["Dt", "se_dyadic"], delta_p=t.loc["Dt", "p_dyadic"],
                             mu=t.loc["Mt", "coef"], mu_se=t.loc["Mt", "se_dyadic"], mu_p=t.loc["Mt", "p_dyadic"],
                             mu_se_2way=t.loc["Mt", "se_twoway"], mu_p_2way=t.loc["Mt", "p_twoway"]))
    return pd.DataFrame(rows)


def outlet_panel(p, slant):
    a = p[["month", "domain", "o1", "o2", "t"] + MEASURES]
    long = pd.concat([a.rename(columns={"o1": "outlet", "o2": "other"}),
                      a.rename(columns={"o2": "outlet", "o1": "other"})])
    iso = long.groupby(["outlet", "month", "domain"], as_index=False).agg(
        **{m: (m, "mean") for m in MEASURES}, t=("t", "first"), n_partners=("other", "size"))
    iso["slant"] = iso.outlet.map(slant)
    iso["slant_c"] = iso.slant - np.mean(list(slant.values()))
    iso["St"] = iso.slant_c * iso.t
    return iso


def isolation_models(iso):
    rows, slopes = [], []
    for dom in DOMAINS:
        x = iso[iso.domain == dom]
        for m in MEASURES:
            r = fe_ols(x, m, ["St"], ["outlet", "month"], cluster="outlet")
            t = r["tab"]
            rows.append(dict(domain=dom, measure=m, lam=t.loc["St", "coef"], se=t.loc["St", "se"],
                             p=t.loc["St", "p"], lo=t.loc["St", "lo"], hi=t.loc["St", "hi"],
                             n=r["n"], n_outlets=r["n_units"]))
            # outlet-specific trends net of month effects
            xx = x.dropna(subset=[m]).copy()
            xx["res"] = xx[m] - xx.groupby("month")[m].transform("mean")
            for o, g in xx.groupby("outlet"):
                b = np.polyfit(g.t, g.res, 1)[0]
                slopes.append(dict(domain=dom, measure=m, outlet=o, slant=g.slant.iloc[0], slope=b,
                                   months=len(g)))
    return pd.DataFrame(rows), pd.DataFrame(slopes)


def plot_isolation(slopes, lam, path):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6), sharex=True)
    for i, dom in enumerate(DOMAINS):
        for j, m in enumerate(MEASURES):
            ax = axes[i, j]
            g = slopes[(slopes.domain == dom) & (slopes.measure == m)]
            ax.axhline(0, color="#555555", lw=0.8)
            ax.scatter(g.slant, g.slope, s=14, color=MEASURE_COL[m], edgecolor="white", lw=0.4, zorder=3)
            b = np.polyfit(g.slant, g.slope, 1)
            xs = np.linspace(g.slant.min(), g.slant.max(), 10)
            ax.plot(xs, np.polyval(b, xs), color="#333333", lw=1, ls="--")
            for _, r in g[g.slant > 3].iterrows():
                dy = -8 if r.outlet == "DailyCaller" else 2
                ax.annotate(SHORT.get(r.outlet, r.outlet),
                            (r.slant, r.slope), fontsize=6, xytext=(3, dy), textcoords="offset points")
            L = lam[(lam.domain == dom) & (lam.measure == m)].iloc[0]
            lam_txt = f"λ = {L.lam:.4f} (SE {L.se:.4f})"
            ax.set_title((MEASURE_LABEL[m] + "\n" if i == 0 else "") + lam_txt, fontsize=7.5 if i else 9)
            if j == 0:
                ax.set_ylabel(f"{DOMAIN_LABEL[dom]}\noutlet trend (per yr)")
            if i == 2:
                ax.set_xlabel("Audience slant (Pew, 1–5)")
    suptitle(fig, "Outlet isolation: trend in each outlet's mean similarity to all others (net of month effects)\n"
                 "vs. audience slant; λ from outlet+month FE model, SEs clustered by outlet",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


# ---------------------------------------------------------------- 3. bloc segregation
def bloc_series(p):
    g = p.groupby(["domain", "month", "within"])[MEASURES].mean().reset_index()
    w = g[g.within == 1].set_index(["domain", "month"])[MEASURES]
    b = g[g.within == 0].set_index(["domain", "month"])[MEASURES]
    seg = (w - b).reset_index()
    per = pd.PeriodIndex(seg.month, freq="M")
    seg["t"] = ((per.year - 2014) * 12 + (per.month - 1)) / 12.0
    return seg


def bloc_trends(seg, p):
    rows = []
    for dom in DOMAINS:
        s = seg[seg.domain == dom].sort_values("month")
        for m in MEASURES:
            X = sm.add_constant(s.t.values)
            ols = sm.OLS(s[m].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            # panel analogue (composition-robust)
            x = p[p.domain == dom].copy()
            x["Bt"] = (1 - x.within) * x.t
            r = fe_ols(x, m, ["Bt"], ["pair", "month"])
            x["LRt"] = (x.bloc_type == "LR") * x.t
            x["RRt"] = (x.bloc_type == "RR") * x.t
            r2 = fe_ols(x, m, ["LRt", "RRt"], ["pair", "month"])
            t2 = r2["tab"]
            rows.append(dict(domain=dom, measure=m,
                             series_slope=ols.params[1], series_se_nw=ols.bse[1], series_p=ols.pvalues[1],
                             phi_between_t=r["tab"].loc["Bt", "coef"], phi_se=r["tab"].loc["Bt", "se_dyadic"],
                             phi_p=r["tab"].loc["Bt", "p_dyadic"],
                             LR_vs_LL=t2.loc["LRt", "coef"], LR_se=t2.loc["LRt", "se_dyadic"], LR_p=t2.loc["LRt", "p_dyadic"],
                             RR_vs_LL=t2.loc["RRt", "coef"], RR_se=t2.loc["RRt", "se_dyadic"], RR_p=t2.loc["RRt", "p_dyadic"]))
    return pd.DataFrame(rows)


def plot_bloc(seg, tr, path, note=""):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    for j, dom in enumerate(DOMAINS):
        ax = axes[j]
        s = seg[seg.domain == dom].sort_values("month")
        x = pd.PeriodIndex(s.month, freq="M").to_timestamp()
        ax.axhline(0, color="#555555", lw=0.8)
        for m in ["tfidf", "sbert"]:
            ax.plot(x, s[m], color=MEASURE_COL[m], lw=1.1, label=MEASURE_LABEL[m])
            T = tr[(tr.domain == dom) & (tr.measure == m)].iloc[0]
            b0 = np.mean(s[m]) - T.series_slope * np.mean(s.t)
            ax.plot(x, b0 + T.series_slope * s.t, color=MEASURE_COL[m], lw=0.9, ls="--")
        ax.set_title(DOMAIN_LABEL[dom])
        if j == 0:
            ax.set_ylabel("Within − between bloc similarity")
        ax.tick_params(axis="x", rotation=0)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    suptitle(fig, "Bloc segregation in coverage similarity (dashed: linear trend)" + note, fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


# ---------------------------------------------------------------- 4. common trend
def common_trend(p):
    out = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            g = month_effects(x, m, ["Dt"])
            g["domain"], g["measure"] = dom, m
            out.append(g)
    return pd.concat(out)


def plot_common(ct, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for j, dom in enumerate(DOMAINS):
        ax = axes[j]
        for m in ["tfidf", "sbert"]:
            g = ct[(ct.domain == dom) & (ct.measure == m)]
            x = pd.PeriodIndex(g.month, freq="M").to_timestamp()
            ax.plot(x, g.gamma, color=MEASURE_COL[m], lw=1.2, label=MEASURE_LABEL[m])
        for d, lab in [("2017-11-01", "280 chars"), ("2020-03-01", "COVID"), ("2022-10-27", "Ownership")]:
            ax.axvline(pd.Timestamp(d), color=OI["grey"], lw=0.7, ls=":")
            if j == 0:
                ax.text(pd.Timestamp(d), 0.02, lab, rotation=90, fontsize=6, transform=ax.get_xaxis_transform(),
                        va="bottom", ha="right", color="#555555")
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(DOMAIN_LABEL[dom])
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        if j == 0:
            ax.set_ylabel("Month effect γₜ (vs. Nov 2015)")
    axes[1].legend(frameon=False, loc="upper right")
    suptitle(fig, "Common trend in pairwise similarity (month FE from Model A; average-distance pair)", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    p, slant = build_panel()
    p = add_terms(p)

    yg = year_gradient(p)
    yg.to_csv(os.path.join(TAB, "year_gradient.csv"), index=False)
    descriptive_gradient(p).to_csv(os.path.join(TAB, "descriptive_gradient_by_year.csv"), index=False)
    plot_year_gradient(yg, os.path.join(FIG, "fig1_year_gradient"))

    ct = common_trend(p)
    ct.to_csv(os.path.join(TAB, "common_trend_gamma.csv"), index=False)
    plot_common(ct, os.path.join(FIG, "fig2_common_trend"))

    seg = bloc_series(p)
    seg.to_csv(os.path.join(TAB, "bloc_segregation_series.csv"), index=False)
    tr = bloc_trends(seg, p)
    tr.to_csv(os.path.join(TAB, "bloc_segregation_trends.csv"), index=False)
    plot_bloc(seg, tr, os.path.join(FIG, "fig3_bloc_segregation"), "\nBlocs: right = slant > 3 (5 outlets)")

    ap = asym_pair(p)
    ap.to_csv(os.path.join(TAB, "asymmetry_midpoint.csv"), index=False)
    iso = outlet_panel(p, slant)
    lam, slopes = isolation_models(iso)
    lam.to_csv(os.path.join(TAB, "isolation_models.csv"), index=False)
    slopes.to_csv(os.path.join(TAB, "isolation_outlet_slopes.csv"), index=False)
    plot_isolation(slopes, lam, os.path.join(FIG, "fig4_isolation"))

    save_fallback_log("extensions")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(yg[yg.measure != "gap"].pivot_table(index="year", columns=["domain", "measure"], values="coef").round(4))
    print(yg[yg.measure == "gap"].pivot_table(index="year", columns=["domain"], values="coef").round(3))
    print(tr.round(5)); print(ap.round(5)); print(lam.round(5))
