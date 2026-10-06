"""
17_network.py -- similarity networks and community structure.

Nodes = 25 outlets. Edge weight = the pair's month-adjusted mean similarity over
the whole window (pair FE from  y = a_pair + g_month [+ post-count terms]; the
count terms are only used for political / non-political, as in 15_postcount_check).
Scores: TF-IDF and centred SBERT (v3, averaged over the five samples).
Weights: both measures centred on the network's average pair (w - mean w) and
the positive part kept, so a tie means "more alike than the average pair".
Without this, raw TF-IDF is a dense all-positive graph dominated by shared
baseline vocabulary and every method returns a single community.

Community detection: Louvain (networkx), 100 seeds, best-modularity partition kept;
stability = mean ARI between seed partitions.

Ideological alignment tests (permutation, 10,000 shuffles of outlet attributes over
the fixed network):
  eta2_slant  share of slant variance explained by community membership
  NMI_bloc    NMI between communities and left/right blocs
  Q_bloc      modularity of the left/right bloc partition (same weights)
  Q_type      modularity of the outlet-type partition (comparison)
Robustness: k-nearest-neighbour backbone (k = 5) instead of the full weighted graph.
Over time: the same statistics on yearly networks.

Output: tables/network_communities.csv, tables/network_tests.csv,
        tables/network_yearly.csv, figures/fig_network.(pdf|png),
        figures/figS_network_yearly.(pdf|png)
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import networkx as nx
from networkx.algorithms import community as nxc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

pc = importlib.import_module("15_postcount_check")
ot = importlib.import_module("08_outlet_type")
hm = importlib.import_module("11_heatmap")
set_style()
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
RNG = np.random.default_rng(20260927)
NPERM, NSEED, KNN = 5000, 100, 5
NQPERM, NYPERM = 1000, 500                         # modularity nulls (whole period / yearly)
if os.environ.get("QUICK") == "1":                 # smoke test (run_pipeline.py --quick)
    NPERM, NSEED, NQPERM, NYPERM = 199, 20, 50, 50
SLANT_CMAP = LinearSegmentedColormap.from_list("slant", ["#1a56a0", "#2a78d6", "#f0efec", "#e34948", "#b8302f"])
SHAPES = ["o", "s", "^", "D", "v", "P"]
LAB = {"tfidf": "Lexical (TF-IDF)", "sbert": "Semantic (centred SBERT)"}
DLAB = {"pol": "Political", "nonpol": "Non-political", "all": "All posts"}


def pair_levels(x, m, ctrl):
    y = x[m].values
    if ctrl:
        g = fe_ols(x, m, ctrl, ["pair", "month"])["tab"].coef
        y = y - sum(g[c] * x[c].values for c in ctrl)
    a = pc.am.pair_effects(x.assign(yy=y), "yy")
    a += x[m].mean() - a.mean()
    k = x.groupby("pair")[["o1", "o2"]].first()
    return k.assign(w=a.reindex(k.index)).reset_index(drop=True)


def graph(levels, measure, knn=None, centre=True):
    G = nx.Graph()
    G.add_nodes_from(set(levels.o1) | set(levels.o2))
    mu = levels.w.mean() if centre else levels.w.min() - 1e-3
    for r in levels.itertuples():
        w = r.w - mu                       # centred: tie = more alike than the average pair
        if w > 0:
            G.add_edge(r.o1, r.o2, weight=w)
    if knn:
        keep = set()
        for n in G.nodes:
            nb = sorted(G[n].items(), key=lambda kv: -kv[1]["weight"])[:knn]
            keep |= {tuple(sorted((n, v))) for v, _ in nb}
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from((a, b, G[a][b]) for a, b in keep)
        G = H
    return G


def louvain(G):
    parts, qs = [], []
    for s in range(NSEED):
        p = nxc.louvain_communities(G, weight="weight", seed=s)
        parts.append(p); qs.append(nxc.modularity(G, p, weight="weight"))
    best = parts[int(np.argmax(qs))]
    nodes = sorted(G.nodes)
    lab = lambda p: [next(i for i, c in enumerate(p) if n in c) for n in nodes]
    L = [lab(p) for p in parts[:100]]
    stab = np.mean([adjusted_rand_score(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))])
    # order communities by mean slant (left first) for stable labelling
    return best, max(qs), stab


def eta2(labels, slant_vals):
    s = pd.Series(slant_vals); g = pd.Series(labels)
    between = sum(len(v) * (v.mean() - s.mean()) ** 2 for _, v in s.groupby(g))
    return between / ((s - s.mean()) ** 2).sum()


def q_of(G, labels, nodes):
    groups = {}
    for n, l in zip(nodes, labels):
        groups.setdefault(l, set()).add(n)
    return nxc.modularity(G, list(groups.values()), weight="weight")


def tests(G, part, slant, cut):
    nodes = sorted(G.nodes)
    com = [next(i for i, c in enumerate(part) if n in c) for n in nodes]
    s = np.array([slant[n] for n in nodes])
    bloc = (s > cut).astype(int)
    typ = [ot.TYPE[n] for n in nodes]
    obs = dict(eta2_slant=eta2(com, s), NMI_bloc=normalized_mutual_info_score(bloc, com),
               ARI_bloc=adjusted_rand_score(bloc, com), NMI_type=normalized_mutual_info_score(typ, com),
               Q_bloc=q_of(G, bloc, nodes), Q_type=q_of(G, typ, nodes))
    null = {k: [] for k in obs}
    for _ in range(NPERM):
        perm = RNG.permutation(len(nodes))
        sp, bp, tp = s[perm], bloc[perm], [typ[i] for i in perm]
        null["eta2_slant"].append(eta2(com, sp))
        null["NMI_bloc"].append(normalized_mutual_info_score(bp, com))
        null["ARI_bloc"].append(adjusted_rand_score(bp, com))
        null["NMI_type"].append(normalized_mutual_info_score(tp, com))
    # modularity nulls are costlier: 1,000 shuffles
    for _ in range(NQPERM):
        perm = RNG.permutation(len(nodes))
        null["Q_bloc"].append(q_of(G, bloc[perm], nodes))
        null["Q_type"].append(q_of(G, [typ[i] for i in perm], nodes))
    out = {}
    for k, v in obs.items():
        v0 = np.array(null[k])
        out[k] = v
        out[k + "_null_mean"] = v0.mean()
        out[k + "_p"] = (1 + (v0 >= v).sum()) / (1 + len(v0))
    return out, dict(zip(nodes, com))


def draw(ax, G, members, slant, title, norm):
    B = graph_backbone = G
    pos = nx.spring_layout(B, weight="weight", seed=7, k=0.9 / np.sqrt(len(B)), iterations=500)
    w = np.array([d["weight"] for _, _, d in B.edges(data=True)])
    wn = (w - w.min()) / (w.max() - w.min() + 1e-12)
    nx.draw_networkx_edges(B, pos, ax=ax, width=0.4 + 2.2 * wn, edge_color="#9a9a9a", alpha=0.55)
    order = sorted(set(members.values()), key=lambda c: np.mean([slant[n] for n in members if members[n] == c]))
    remap = {c: i for i, c in enumerate(order)}
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=150, marker=SHAPES[remap[members[n]] % len(SHAPES)], c=[SLANT_CMAP(norm(slant[n]))],
                   edgecolors="#2b2b2b", linewidths=0.8, zorder=3)
        ax.annotate(hm.NAMES[n], (x, y), xytext=(0, 8.5), textcoords="offset points", ha="center",
                    fontsize=6.2, color="#222222", zorder=4)
    ax.set_title(title, fontsize=8.5)
    ax.set_axis_off()
    ax.margins(0.12)
    return remap


if __name__ == "__main__":
    common.DATA_SIM = AVG
    comm_rows, test_rows, year_rows, drawn = [], [], [], {}
    for src in ["pew", "adfontes"]:
        common.SLANT_SOURCE = src
        p, slant = build_panel()
        p = pc.add_counts(p)
        cut = BLOC_CUT[src]
        for dom in ["pol", "nonpol", "all"]:
            x = p[p.domain == dom]
            ctrl = [] if dom == "all" else pc.CTRL
            for m in ["tfidf", "sbert"]:
                lev = pair_levels(x, m, ctrl)
                for variant, knn in [("full weighted", None), (f"kNN backbone (k={KNN})", KNN)]:
                    G = graph(lev, m, knn)
                    part, q, stab = louvain(G)
                    res, members = tests(G, part, slant, cut)
                    test_rows.append(dict(source=src, domain=dom, measure=m, graph=variant, n_communities=len(part),
                                          Q_louvain=q, stability_ARI=stab, **res))
                    if src == "pew":
                        for n, c in members.items():
                            comm_rows.append(dict(domain=dom, measure=m, graph=variant, outlet=n, community=c,
                                                  slant_pew=slant[n], type=ot.TYPE[n]))
                        if variant.startswith("full"):
                            drawn[(dom, m)] = (graph(lev, m, KNN, centre=False), members, dict(slant))
                # yearly networks (full weighted), Pew and Ad Fontes
                for yr in range(2015, 2023):
                    xy = x[x.year.clip(2015, 2022) == yr]
                    Gy = graph(pair_levels(xy, m, ctrl), m)
                    nodes = sorted(Gy.nodes)
                    s = np.array([slant[n] for n in nodes]); bloc = (s > cut).astype(int)
                    best = max((nxc.louvain_communities(Gy, weight="weight", seed=sd) for sd in range(20)),
                               key=lambda pp: nxc.modularity(Gy, pp, weight="weight"))
                    com = [next(i for i, c in enumerate(best) if n in c) for n in nodes]
                    qb = q_of(Gy, bloc, nodes)
                    qnull = np.array([q_of(Gy, bloc[RNG.permutation(len(nodes))], nodes) for _ in range(NYPERM)])
                    year_rows.append(dict(source=src, domain=dom, measure=m, year=yr, n_nodes=len(nodes),
                                          n_communities=len(best), Q_bloc=qb, Q_bloc_p=(1 + (qnull >= qb).sum()) / (1 + len(qnull)),
                                          eta2_slant=eta2(com, s), NMI_bloc=normalized_mutual_info_score(bloc, com)))
            print(f"done {src} {dom}", flush=True)
    comm, tst, yrs = pd.DataFrame(comm_rows), pd.DataFrame(test_rows), pd.DataFrame(year_rows)
    comm.to_csv(os.path.join(TAB, "network_communities.csv"), index=False)
    tst.to_csv(os.path.join(TAB, "network_tests.csv"), index=False)
    yrs.to_csv(os.path.join(TAB, "network_yearly.csv"), index=False)

    # ---- figure: kNN backbones, political and non-political, both measures (Pew slant)
    common.SLANT_SOURCE = "pew"
    _, slant = build_panel()
    vals = np.array(list(slant.values()))
    norm = TwoSlopeNorm(vcenter=3.0, vmin=vals.min(), vmax=vals.max())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))
    for i, dom in enumerate(["pol", "nonpol"]):
        for j, m in enumerate(["tfidf", "sbert"]):
            G, members, sl = drawn[(dom, m)]
            draw(axes[i, j], G, members, sl, f"{DLAB[dom]}: {LAB[m]}", norm)
    sm_ = plt.cm.ScalarMappable(cmap=SLANT_CMAP, norm=norm)
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.013])
    cb = fig.colorbar(sm_, cax=cax, orientation="horizontal")
    cb.set_label("Pew audience slant (fill colour; 3 = midpoint)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker=SHAPES[k], ls="", mfc="white", mec="#2b2b2b", ms=7,
                      label=f"Community {k + 1}") for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.075), title="Marker shape = Louvain community (numbered left to right by mean slant)",
               title_fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = os.path.join(FIG, "fig_network")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200); plt.close(fig)

    # ---- figure: yearly bloc modularity and slant eta2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, stat, ylab in [(axes[0], "Q_bloc", "Modularity of left/right split"),
                           (axes[1], "eta2_slant", "Slant variance explained by communities")]:
        for dom, ls in [("pol", "-"), ("nonpol", "--")]:
            for m in ["tfidf", "sbert"]:
                g = yrs[(yrs.source == "pew") & (yrs.domain == dom) & (yrs.measure == m)]
                ax.plot(g.year, g[stat], ls=ls, lw=2, color=MEASURE_COL[m], marker="o", ms=4,
                        label=f"{DLAB[dom]}, {LAB[m].split(' (')[0].lower()}")
        ax.set_ylabel(ylab, fontsize=8)
        ax.axhline(0, color="#777777", lw=0.6)
    axes[0].legend(frameon=False, fontsize=6.8, loc="best")
    fig.tight_layout()
    path = os.path.join(FIG, "figS_network_yearly")
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200); plt.close(fig)
    save_fallback_log("network")

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    cols = ["source", "domain", "measure", "graph", "n_communities", "Q_louvain", "stability_ARI",
            "eta2_slant", "eta2_slant_p", "NMI_bloc", "NMI_bloc_p", "Q_bloc", "Q_bloc_p", "NMI_type", "NMI_type_p", "Q_type", "Q_type_p"]
    print(tst[cols].round(3).to_string(index=False))
    for (dom, m, gr), g in comm[comm.graph.str.startswith("full")].groupby(["domain", "measure", "graph"]):
        print(f"\n{dom} {m} {gr}")
        for c, h in g.groupby("community"):
            print(f"  C{c} (mean slant {h.slant_pew.mean():.2f}): " + ", ".join(hm.NAMES[o] for o in h.sort_values("slant_pew").outlet))
    print(yrs.pivot_table(index=["source", "domain", "measure"], columns="year", values="Q_bloc").round(3).to_string())
