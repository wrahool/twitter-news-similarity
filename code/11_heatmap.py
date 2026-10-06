"""
11_heatmap.py -- pairwise similarity heatmaps, outlets ordered by audience slant.

x axis: outlets ordered left -> right by slant; y axis: same order bottom -> top.
Lower triangle only (each pair once). Cell colour = the pair's mean similarity
over Nov 2015 - Jan 2023 (scores averaged over the five samples), on a
diverging blue-grey-red scale centred at the median pair (blue = more similar
than the median pair, red = less similar). Separate colour scales per panel.
Dashed lines mark the boundary of the right-leaning bloc (slant > 3).

Outputs: figures/fig_heatmap_political.(pdf|png)   (TF-IDF, SBERT; political)
         figures/figS9_heatmap_all_domains.(pdf|png) (3 domains x 2 measures)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

set_style()
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
NAMES = {"guardian": "Guardian", "HuffPost": "HuffPost", "NPR": "NPR", "VICE": "VICE", "nytimes": "NYT",
         "washingtonpost": "WaPo", "politico": "Politico", "BBCWorld": "BBC World", "BuzzFeedNews": "BuzzFeed",
         "MSNBC": "MSNBC", "Newsweek": "Newsweek", "thehill": "The Hill", "TIME": "TIME",
         "BusinessInsider": "Bus. Insider", "CNN": "CNN", "NBCNews": "NBC", "WSJ": "WSJ", "CBSNews": "CBS",
         "USATODAY": "USA Today", "ABC": "ABC", "nypost": "NY Post", "dcexaminer": "Wash. Examiner",
         "FoxNews": "Fox News", "DailyCaller": "Daily Caller", "BreitbartNews": "Breitbart"}
# diverging pair from the dataviz reference palette: red (low) - neutral grey - blue (high)
CMAP = LinearSegmentedColormap.from_list("div", ["#b8302f", "#e34948", "#f0efec", "#2a78d6", "#1a56a0"])


def matrix(pm, m, order):
    idx = {o: k for k, o in enumerate(order)}
    M = np.full((len(order), len(order)), np.nan)
    for _, r in pm.iterrows():
        a, b = idx[r.o1], idx[r.o2]
        i, j = max(a, b), min(a, b)          # x = i (more right), y = j  -> lower triangle
        M[j, i] = r[m]                       # row = y, col = x
    return M


def draw(ax, M, order, slant, title, cbar_label, fig):
    vals = M[~np.isnan(M)]
    norm = TwoSlopeNorm(vcenter=np.median(vals), vmin=vals.min(), vmax=vals.max())
    im = ax.pcolormesh(np.arange(len(order) + 1), np.arange(len(order) + 1), np.ma.masked_invalid(M),
                       cmap=CMAP, norm=norm, edgecolors="#fcfcfb", linewidth=0.6)
    n = len(order)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([f"{NAMES[o]} ({slant[o]:.2f})" for o in order], rotation=90, fontsize=6.3)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([f"{NAMES[o]} ({slant[o]:.2f})" for o in order], fontsize=6.3)
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    k = sum(slant[o] <= 3 for o in order)                    # first right-bloc index
    ax.plot([k, k], [0, k], color="#333333", lw=0.9, ls="--")
    ax.plot([k, n], [k, k], color="#333333", lw=0.9, ls="--")
    ax.set_title(title, fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.75)
    cb.set_label(cbar_label, fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.ax.axhline(np.median(vals), color="#333333", lw=0.8)


if __name__ == "__main__":
    common.DATA_SIM = AVG
    p, slant = build_panel()
    order = sorted(slant, key=lambda o: (slant[o], o))
    pm = p.groupby(["domain", "o1", "o2"])[["tfidf", "sbert"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, m in zip(axes, ["tfidf", "sbert"]):
        M = matrix(pm[pm.domain == "pol"], m, order)
        draw(ax, M, order, slant, f"Political content: {MEASURE_LABEL[m]}",
             "Mean similarity (line = median pair)", fig)
    fig.tight_layout()
    path = os.path.join(FIG, "fig_heatmap_political")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 16.5))
    for i, d in enumerate(DOMAINS):
        for j, m in enumerate(["tfidf", "sbert"]):
            draw(axes[i, j], matrix(pm[pm.domain == d], m, order), order, slant,
                 f"{DOMAIN_LABEL[d]}: {MEASURE_LABEL[m]}", "Mean similarity (line = median pair)", fig)
    fig.tight_layout()
    path = os.path.join(FIG, "figS9_heatmap_all_domains")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)
    print("done")
