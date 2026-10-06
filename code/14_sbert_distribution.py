"""
14_sbert_distribution.py -- distribution of SBERT similarity scores.

Pair-month SBERT scores (average of the five samples; Nov 2015 - Jan 2023;
25,025 pair-months per domain), by domain, with the distribution of pair means
(300 pairs, averaged over months) overlaid. Also writes summary statistics.

Outputs: figures/figS11_sbert_distribution.(pdf|png), tables/sbert_distribution.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt

set_style()
common.DATA_SIM = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
p, _ = build_panel()
rows = []
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=False)
bins = np.linspace(0.4, 1.0, 61)
for ax, d in zip(axes, DOMAINS):
    x = p[p.domain == d]
    v = x.sbert.values
    pm = x.groupby("pair").sbert.mean().values
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    rows.append(dict(domain=d, level="pair-month", n=len(v), mean=v.mean(), sd=v.std(), min=q[0], p1=q[1], p5=q[2],
                     p25=q[3], median=q[4], p75=q[5], p95=q[6], p99=q[7], max=q[8],
                     share_above_0_9=(v > .9).mean(), share_above_0_8=(v > .8).mean(),
                     sd_within_month=x.groupby("month").sbert.std().mean()))
    qp = np.percentile(pm, [0, 25, 50, 75, 100])
    rows.append(dict(domain=d, level="pair mean", n=len(pm), mean=pm.mean(), sd=pm.std(), min=qp[0], p25=qp[1],
                     median=qp[2], p75=qp[3], max=qp[4], share_above_0_9=(pm > .9).mean(),
                     share_above_0_8=(pm > .8).mean()))
    ax.hist(v, bins=bins, color=OI["orange"], alpha=0.85, density=True, label="Pair-months")
    ax.hist(pm, bins=bins, histtype="step", color="#333333", lw=1.1, density=True, label="Pair means")
    ax.axvline(np.median(v), color="#333333", lw=0.8, ls="--")
    ax.set_title(f"{DOMAIN_LABEL[d]}\nmedian {np.median(v):.3f}, IQR {q[3]:.3f}–{q[5]:.3f}", fontsize=8.5)
    ax.set_xlabel("SBERT cosine similarity")
    ax.set_xlim(0.4, 1.0)
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
path = os.path.join(FIG, "figS11_sbert_distribution")
fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB, "sbert_distribution.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.round(3).to_string(index=False))
