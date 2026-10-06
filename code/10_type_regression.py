"""
10_type_regression.py -- one cross-sectional model per domain x measure:

  similarity_ij = b0 + b1*D_ij + sum_k c_k * n_ijk + e_ij

where n_ijk (0, 1 or 2) is the number of outlets of type k in pair ij
(types: broadcast, newspaper, magazine, digital-first, UK-based; see
08_outlet_type.py). Because the counts always sum to 2, the five type terms
plus an intercept are not separately identified; we use effect (sum-to-zero)
coding, so each c_k is the change in expected similarity per outlet of type k
relative to the average type, and all five are reported. Expected similarity
of a pair = b0 + b1*D + c_type(i) + c_type(j).

Unit: one row per pair; outcome = the pair's mean similarity over Nov 2015 -
Jan 2023, using scores averaged over the five samples. SEs: dyadic
cluster-robust (Aronow et al. 2015), t with 24 df.

Outputs: tables/type_regression.csv, figures/figS8_type_regression.(pdf|png)
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

set_style()
TYPE = importlib.import_module("08_outlet_type").TYPE
TYPES = ["broadcast", "newspaper", "magazine", "digital", "uk"]
TLAB = {"broadcast": "Broadcast", "newspaper": "Newspaper", "magazine": "Magazine",
        "digital": "Digital-first", "uk": "UK-based"}
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
TCRIT = stats.t.ppf(.975, 24)


def fit(df, y):
    n = {k: (df.o1.map(TYPE) == k).astype(float) + (df.o2.map(TYPE) == k).astype(float) for k in TYPES}
    ref = TYPES[-1]
    X = np.column_stack([np.ones(len(df)), df.D.values] + [n[k] - n[ref] for k in TYPES[:-1]])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ df[y].values
    e = df[y].values - X @ b
    V, G = vcov_dyadic(X, e, df.o1.values, df.o2.values, df.pair.values, XtX_inv)
    fix = vcov_dyadic.last_fallback
    # recover the reference-type effect: c_ref = -sum(c_k), with its SE
    L = np.zeros((len(TYPES), X.shape[1]))
    for j in range(len(TYPES) - 1):
        L[j, 2 + j] = 1
    L[-1, 2:] = -1
    est = np.r_[b[0], b[1], L @ b]
    A = np.vstack([np.eye(X.shape[1])[:2], L])
    cov = A @ V @ A.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    r2 = 1 - (e @ e) / ((df[y] - df[y].mean()) ** 2).sum()
    names = ["Intercept (b0)", "Ideological distance D (b1)"] + [TLAB[k] for k in TYPES]
    out = pd.DataFrame({"term": names, "est": est, "se": se})
    out["lo"], out["hi"] = out.est - TCRIT * out.se, out.est + TCRIT * out.se
    out["p"] = 2 * stats.t.sf(np.abs(out.est / out.se), G - 1)
    out["r2"] = r2
    out["vcov_fix"] = fix
    return out


def plot(res, path):
    terms = ["Ideological distance D (b1)"] + [TLAB[k] for k in TYPES]
    dcol = {"pol": OI["vermillion"], "nonpol": OI["blue"], "all": OI["grey"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    offs = {"pol": 0.22, "nonpol": 0.0, "all": -0.22}
    for ax, m in zip(axes, ["tfidf", "sbert"]):
        for i, term in enumerate(terms[::-1]):
            for d in DOMAINS:
                r = res[(res.measure == m) & (res.domain == d) & (res.term == term)].iloc[0]
                ax.errorbar(r.est, i + offs[d], xerr=TCRIT * r.se, fmt="o", ms=4.2, color=dcol[d],
                            elinewidth=1.3, capsize=2.5, mfc=dcol[d] if r.p < .05 else "white", mew=1.2)
            if i == len(terms) - 1:
                ax.axhspan(i - 0.5, i + 0.5, color="#f1f1f1", zorder=0)
        ax.axvline(0, color="#444444", lw=0.9)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(["Ideological distance D\n(per slant point)" if "distance" in t else f"{t}\n(per outlet)"
                            for t in terms[::-1]])
        ax.set_ylim(-0.6, len(terms) - 0.4)
        ax.grid(axis="y", visible=False)
        r2s = ", ".join(f"{DOMAIN_LABEL[d].split()[0]} {res[(res.measure == m) & (res.domain == d)].r2.iloc[0]:.2f}"
                        for d in DOMAINS)
        ax.set_title(f"{MEASURE_LABEL[m]}\n$R^2$: {r2s}", fontsize=8.5)
        ax.set_xlabel("Effect on pairwise similarity")
    h = [Line2D([0], [0], marker="o", color=dcol[d], lw=1.3, ms=4.5, label=DOMAIN_LABEL[d]) for d in DOMAINS]
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png"); plt.close(fig)


if __name__ == "__main__":
    common.DATA_SIM = AVG
    p, _ = build_panel()
    pm = p.groupby(["domain", "pair", "o1", "o2", "D"])[["tfidf", "sbert"]].mean().reset_index()
    res = []
    for d in DOMAINS:
        for m in ["tfidf", "sbert"]:
            r = fit(pm[pm.domain == d].reset_index(drop=True), m)
            r.insert(0, "measure", m); r.insert(0, "domain", d)
            res.append(r)
    res = pd.concat(res)
    res.to_csv(os.path.join(TAB, "type_regression.csv"), index=False)
    plot(res, os.path.join(FIG, "figS8_type_regression"))
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
