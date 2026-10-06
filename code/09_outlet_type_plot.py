"""
09_outlet_type_plot.py -- coefficient (whisker) plot for the outlet-type checks
produced by 08_outlet_type.py.

Left:  cross-sectional slope of pair-mean similarity on ideological distance D
       under four specifications.
Right: panel distance-gradient trend delta (Model A) under three specifications.
Points = estimates; whiskers = 95% CIs from dyadic cluster-robust SEs (t, 24 df).
The gap measure G is omitted (different scale).

Output: figures/figS7_outlet_type_coefplot.(pdf|png)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

set_style()
TCRIT = stats.t.ppf(.975, 24)
ROWS = [(d, m) for d in DOMAINS for m in ["tfidf", "sbert"]]
ROWLAB = [f"{DOMAIN_LABEL[d]}, {MEASURE_LABEL[m]}" for d, m in ROWS]


def panel(ax, df, specs, colors, title, xlabel):
    k = len(specs)
    offs = np.linspace(-0.27, 0.27, k)
    for i, (d, m) in enumerate(ROWS[::-1]):
        for j, s in enumerate(specs):
            r = df[(df.domain == d) & (df.measure == m) & (df.model == s)].iloc[0]
            est = r.slope_D if "slope_D" in r else r.delta
            y = i + offs[::-1][j]
            ax.errorbar(est, y, xerr=TCRIT * r.se, fmt="o", ms=4.2, color=colors[j], ecolor=colors[j],
                        elinewidth=1.3, capsize=2.5, mfc=colors[j] if r.p < .05 else "white", mew=1.2)
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f4f4", zorder=0)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWLAB[::-1])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    handles = [Line2D([0], [0], marker="o", color=c, lw=1.3, ms=4.5, label=s) for s, c in zip(specs, colors)]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)


if __name__ == "__main__":
    xs = pd.read_csv(os.path.join(TAB, "outlet_type_cross_section.csv"))
    pn = pd.read_csv(os.path.join(TAB, "outlet_type_panel.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), sharey=True)
    panel(axes[0], xs, ["(a) D only", "(b) + same type", "(c) + type-pair FE", "(d) + outlet FE"],
          [OI["grey"], OI["sky"], OI["blue"], OI["vermillion"]],
          "A. Level: similarity vs. distance (pair means)", "Slope on $D$ (similarity per slant point)")
    panel(axes[1], pn, ["Model A", "+ type-pair x t", "+ outlet x t"],
          [OI["grey"], OI["blue"], OI["vermillion"]],
          "B. Change: distance-gradient trend $\\delta$", "$\\delta$ ($D \\times t$, per slant point per year)")
    fig.text(0.5, -0.1, "Points: estimates (filled = p < .05). Whiskers: 95% CIs, dyadic cluster-robust SEs. "
             "Scores averaged over five samples; Nov 2015–Jan 2023.", ha="center", fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIG, "figS7_outlet_type_coefplot")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)
    print("saved", path)
