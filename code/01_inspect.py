"""
01_inspect.py -- Step 1: data inspection. Writes tables/inspection_*.csv,
tables/inspection_report.md and figures/figS0_distributions.(pdf|png) +
figures/figS0b_monthly_means.(pdf|png) (the latter shows the Oct-Nov 2015
TF-IDF break that motivates the 2015-11 start of the primary window).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib.pyplot as plt

set_style()
COLS = ["TF-IDF-Pol", "SBERT-Pol", "TF-IDF-NonPol", "SBERT-NonPol", "TF-IDF-All", "SBERT-All"]

if __name__ == "__main__":
    d, s = load_raw()
    L = []
    L.append(f"Rows: {len(d)}; columns: {list(d.columns)}")
    outlets = sorted(set(d["Outlet 1"]) | set(d["Outlet 2"]))
    L.append(f"Outlets: {len(outlets)}; unordered pairs observed: {d.groupby(['Outlet 1','Outlet 2']).ngroups}")
    L.append(f"Months: {d.Month.min()} to {d.Month.max()} ({d.Month.nunique()} months)")
    L.append(f"'Sample' column values: {sorted(d.Sample.unique())}")
    key = d.apply(lambda r: "|".join(sorted([r["Outlet 1"], r["Outlet 2"]])), axis=1) + d.Month
    L.append(f"Pairs stored once? {key.nunique() == len(d)} (unique unordered pair-months = {key.nunique()}); "
             f"self-pairs: {(d['Outlet 1'] == d['Outlet 2']).sum()}")
    pos = {m: i for i, m in enumerate(s.Media)}
    L.append(f"Outlet 1 precedes Outlet 2 in slant-file order: {(d['Outlet 1'].map(pos) < d['Outlet 2'].map(pos)).mean():.0%}")
    L.append(f"Names only in similarity file: {sorted(set(outlets) - set(s.Media))}; "
             f"only in slant file: {sorted(set(s.Media) - set(outlets))}")
    L.append(f"NA values: {int(d[COLS].isna().sum().sum())}; values outside [0,1]: "
             f"{int(((d[COLS] < 0) | (d[COLS] > 1)).sum().sum())}")
    desc = d[COLS].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc.to_csv(os.path.join(TAB, "inspection_distributions.csv"))

    # Missingness by outlet-month
    months = pd.period_range(d.Month.min(), d.Month.max(), freq="M").astype(str)
    long = pd.concat([d[["Month", "Outlet 1"]].set_axis(["month", "o"], axis=1),
                      d[["Month", "Outlet 2"]].set_axis(["month", "o"], axis=1)])
    pres = long.drop_duplicates().assign(v=1).pivot(index="o", columns="month", values="v") \
        .reindex(index=s.Media, columns=months).fillna(0).astype(bool)
    gaps = []
    for o in s.Media:
        miss = [m for m in months if not pres.loc[o, m]]
        if not miss:
            continue
        run = [miss[0]]
        for m in miss[1:]:
            if (pd.Period(m, "M") - pd.Period(run[-1], "M")).n == 1:
                run.append(m)
            else:
                gaps.append((o, run[0], run[-1], len(run))); run = [m]
        gaps.append((o, run[0], run[-1], len(run)))
    gaps = pd.DataFrame(gaps, columns=["outlet", "start", "end", "months"])
    gaps.to_csv(os.path.join(TAB, "inspection_gaps.csv"), index=False)
    # Are gaps whole outlet-months (all pairs missing) or partial?
    n_present = pres.sum()
    exp = (n_present * (n_present - 1) / 2).astype(int)
    obs = d.groupby("Month").size().reindex(months)
    L.append(f"Rows per month equal C(n_present, 2) in every month: {bool((exp == obs).all())} "
             "(i.e. gaps are whole outlet-months, no partial pair gaps)")
    L.append(f"Slant: min {s.slant.min()} ({s.Media[s.slant.idxmin()]}), max {s.slant.max()} "
             f"({s.Media[s.slant.idxmax()]}), median {s.slant.median()}; outlets > 3: "
             f"{list(s.Media[s.slant > 3])}")
    mm = d.groupby("Month")[COLS].agg(["mean", "std"])
    mm.to_csv(os.path.join(TAB, "inspection_monthly_means.csv"))
    L.append("TF-IDF break: mean / cross-pair SD of TF-IDF-All by month, 2015-08..2015-12: " +
             "; ".join(f"{m}: {mm.loc[m, ('TF-IDF-All','mean')]:.3f} / {mm.loc[m, ('TF-IDF-All','std')]:.3f}"
                       for m in ["2015-08", "2015-09", "2015-10", "2015-11", "2015-12"]))
    with open(os.path.join(TAB, "inspection_report.md"), "w") as f:
        f.write("# Data inspection\n\n" + "\n".join(f"- {x}" for x in L) +
                "\n\n## Gaps (outlet absent from every pair)\n\n" + gaps.to_markdown(index=False) +
                "\n\n## Distributions\n\n" + desc.round(3).to_markdown())
    print("\n".join(L)); print(gaps)

    # Figures
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    for k, c in enumerate(COLS):
        ax = axes[k % 2, k // 2]
        pre = d[d.Month < PRIMARY_START][c]; post = d[d.Month >= PRIMARY_START][c]
        bins = np.linspace(0.4, 1.0, 61)
        ax.hist(post, bins=bins, color=OI["blue"] if "TF" in c else OI["orange"], alpha=0.85, label="2015-11 on")
        ax.hist(pre, bins=bins, histtype="step", color="#333333", lw=0.9, label="before 2015-11")
        ax.set_title(c)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    suptitle(fig, "Distribution of pairwise similarity scores", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS0_distributions.pdf"))
    fig.savefig(os.path.join(FIG, "figS0_distributions.png")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = pd.PeriodIndex(mm.index, freq="M").to_timestamp()
    for c, ls in [("Pol", "-"), ("NonPol", "--"), ("All", ":")]:
        axes[0].plot(x, mm[(f"TF-IDF-{c}", "mean")], ls, color=OI["blue"], lw=1, label=f"TF-IDF-{c}")
        axes[0].plot(x, mm[(f"SBERT-{c}", "mean")], ls, color=OI["orange"], lw=1, label=f"SBERT-{c}")
        axes[1].plot(x, mm[(f"TF-IDF-{c}", "std")], ls, color=OI["blue"], lw=1)
        axes[1].plot(x, mm[(f"SBERT-{c}", "std")], ls, color=OI["orange"], lw=1)
    for ax in axes:
        ax.axvline(pd.Timestamp("2015-11-01"), color=OI["vermillion"], lw=0.8)
    axes[0].set_title("Monthly mean across pairs"); axes[1].set_title("Monthly cross-pair SD")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    suptitle(fig, "Raw series; red line = start of primary window (TF-IDF regime break, Oct–Nov 2015)", fontsize=9, y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS0b_monthly_means.pdf"))
    fig.savefig(os.path.join(FIG, "figS0b_monthly_means.png")); plt.close(fig)
