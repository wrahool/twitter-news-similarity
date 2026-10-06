"""
19_network_compare.py -- did the similarity network change between the start and
the end of the period, and who moved?

Periods (defaults): A = Nov 2014 - Oct 2017, B = Jan 2021 - Jan 2023. Both are
outside Fox News's 2018-20 data gap. Ties = month-adjusted pair means within each
period (pair FE from y = a_pair + g_month, net of the post-count terms for
political / non-political, coefficients estimated once on the full window).
Measures: TF-IDF and centred SBERT (v3 averaged scores). Domains: political,
non-political, all.

Analyses
 1. QAP correlation of the A and B tie matrices (outlet-permutation p): how much
    of the structure persisted.
 2. MRQAP of the change in ties (B - A) on ideological distance, both-right and
    one-right bloc dummies, and same outlet type; p-values by Dekker, Krackhardt &
    Snijders (2007) double semi-partialling (outlet permutations).
 3. Outlet-level change. Uncertainty from a moving-block bootstrap of months
    within each period (--block months per block; month effects re-estimated in
    every replicate). Six years apart, nearly every outlet's ties change somewhat,
    so the informative test is whether an outlet moved MORE than the typical
    outlet (difference from the median outlet, bootstrap p, BH q across outlets):
      profile_r   correlation of the outlet's ties to the other 24 in A vs B
                  (low = its position changed)
      drift_right change in (mean tie to right-bloc outlets - mean tie to left-bloc
                  outlets), B minus A; > 0 = moved towards the right bloc
    (A month-label permutation null was tried and rejected: shuffled pseudo-periods
    mix both eras, so it flags any slow drift and marks every outlet as changed.)
 4. Communities: consensus Louvain partition in each period (co-assignment over
    --n-seeds runs, Lancichinetti & Fortunato 2012 consensus). ARI / NMI between
    A and B, benchmarked against the agreement between each period's partition
    and its own bootstrap replicates (the sampling-noise ceiling). Ideological
    alignment per period (eta^2 of slant, NMI with blocs) and its change, with
    bootstrap CIs (lighter consensus, --n-seeds-boot runs, inside replicates).
 5. Multilayer modularity (Mucha et al. 2010) with leidenalg: the two periods as
    layers coupled by identity links of weight omega (RB-configuration, i.e. unnormalised,
    modularity with resolution 1, so omega and ties share a scale). Tie weights are rescaled so
    the mean positive tie in each layer is 1, putting omega on the scale of a
    typical tie. Per omega: best partition, who switches, and the share of runs
    in which each outlet switches.

Usage:
    python 19_network_compare.py                 # defaults: 999 bootstrap replicates
    python 19_network_compare.py --quick         # 49 replicates, smoke test
    python 19_network_compare.py --period-a 2014-11:2017-10 --period-b 2021-01:2023-01 --source adfontes
Outputs: tables/netcmp_*.csv, figures/fig_network_change.(pdf|png),
         figures/figS_network_alluvial.(pdf|png)
"""
import os, sys, time, argparse, importlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
import networkx as nx
from networkx.algorithms import community as nxc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed

ri = importlib.import_module("18_robust_inference")
ot = importlib.import_module("08_outlet_type")
hm = importlib.import_module("11_heatmap")
DLAB = {"pol": "Political", "nonpol": "Non-political", "all": "All posts"}
MLAB = {"tfidf": "Lexical (TF-IDF)", "sbert": "Semantic (centred SBERT)"}


# ---------------------------------------------------------------------------
# tie matrices
# ---------------------------------------------------------------------------
def adjusted_scores(p, domain, measure):
    """Scores net of post-count terms (pol / nonpol), coefficients from the full window."""
    x = p[p.domain == domain].reset_index(drop=True)
    y = x[measure].values.astype(float)
    if domain != "all":
        g = fe_ols(x, measure, ri.CTRL, ["pair", "month"])["tab"].coef
        y = y - sum(g[c] * x[c].values for c in ri.CTRL)
    return pd.DataFrame({"month": x.month.values, "pair": x.pair.values, "o1": x.o1.values,
                         "o2": x.o2.values, "y": y})


def pair_fe(x, tol=1e-10, maxit=5000):
    """Pair effects from y = a_pair + g_month (alternating means), mean-centred."""
    pc, pairs = pd.factorize(x.pair)
    mc, _ = pd.factorize(x.month)
    v = x.y.values
    npc, nmc = np.bincount(pc), np.bincount(mc)
    a = np.bincount(pc, weights=v) / npc
    for _ in range(maxit):
        g = np.bincount(mc, weights=v - a[pc]) / nmc
        a_new = np.bincount(pc, weights=v - g[mc]) / npc
        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a = a_new
    return pd.Series(a - a.mean(), index=pairs)


def tie_matrix(x, months, outlets, keys):
    """outlets x outlets matrix of pair levels over the given months (NaN diagonal / missing)."""
    a = pair_fe(x if months is None else x[x.month.isin(months)])
    idx = {o: i for i, o in enumerate(outlets)}
    W = np.full((len(outlets), len(outlets)), np.nan)
    k = keys.loc[keys.index.intersection(a.index)]
    i, j = k.o1.map(idx).values, k.o2.map(idx).values
    W[i, j] = W[j, i] = a.reindex(k.index).values
    return W


def upper(W):
    iu = np.triu_indices(W.shape[0], 1)
    return W[iu]


# ---------------------------------------------------------------------------
# QAP / MRQAP
# ---------------------------------------------------------------------------
def qap_cor(A, B, n, rng):
    a, b = upper(A), upper(B)
    ok = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    null = []
    for _ in range(n):
        pi = rng.permutation(A.shape[0])
        bp = upper(B[np.ix_(pi, pi)])
        ok2 = np.isfinite(a) & np.isfinite(bp)
        null.append(np.corrcoef(a[ok2], bp[ok2])[0, 1])
    null = np.array(null)
    return r, (1 + np.sum(null >= r)) / (1 + n)


def _ols_t(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    return b, b / np.sqrt(np.diag(XtX_inv) * s2)


def mrqap_dsp(Y, Xs, names, n, rng):
    """Dekker et al. (2007) double semi-partialling MRQAP. Y, Xs: square matrices."""
    y = upper(Y)
    cols = [upper(X) for X in Xs]
    ok = np.isfinite(y) & np.all([np.isfinite(c) for c in cols], axis=0)
    X = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in cols])
    b, t = _ols_t(y[ok], X)
    rows = []
    N = Y.shape[0]
    iu = np.triu_indices(N, 1)
    for k, name in enumerate(names):
        others = np.delete(X, k + 1, axis=1)
        bk = np.linalg.lstsq(others, X[:, k + 1], rcond=None)[0]
        E = np.full((N, N), np.nan)
        res = X[:, k + 1] - others @ bk
        tmp = np.full(len(iu[0]), np.nan)
        tmp[ok] = res
        E[iu] = tmp
        E[(iu[1], iu[0])] = tmp
        null = []
        for _ in range(n):
            pi = rng.permutation(N)
            ep = upper(E[np.ix_(pi, pi)])
            ok2 = np.isfinite(ep) & ok
            yy = y[ok2]
            XX = np.column_stack([np.ones(ok2.sum())] + [c[ok2] for c in cols])
            XX[:, k + 1] = ep[ok2]
            null.append(_ols_t(yy, XX)[1][k + 1])
        null = np.array(null)
        rows.append(dict(term=name, coef=b[k + 1], t=t[k + 1], p_dsp=(1 + np.sum(np.abs(null) >= abs(t[k + 1]))) / (1 + n)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# communities
# ---------------------------------------------------------------------------
def centred_graph(W, outlets):
    """Positive part of (tie - mean tie): a tie = more alike than the average pair."""
    G = nx.Graph()
    G.add_nodes_from(outlets)
    mu = np.nanmean(upper(W))
    N = len(outlets)
    for i in range(N):
        for j in range(i + 1, N):
            w = W[i, j] - mu
            if np.isfinite(w) and w > 0:
                G.add_edge(outlets[i], outlets[j], weight=float(w))
    return G


def labels(part, outlets):
    lab = {}
    for c, nodes in enumerate(part):
        for n_ in nodes:
            lab[n_] = c
    return np.array([lab[o] for o in outlets])


def consensus(G, outlets, n_seeds, seed0=0, tau=0.5, max_rounds=10):
    """Lancichinetti & Fortunato (2012) consensus clustering with Louvain."""
    N = len(outlets)
    H = G
    for rnd in range(max_rounds):
        C = np.zeros((N, N))
        for s in range(n_seeds):
            lab = labels(nxc.louvain_communities(H, weight="weight", seed=seed0 + 1000 * rnd + s), outlets)
            C += lab[:, None] == lab[None, :]
        C /= n_seeds
        if np.all((C < 1e-9) | (C > 1 - 1e-9)):
            break
        H = nx.Graph()
        H.add_nodes_from(outlets)
        for i in range(N):
            for j in range(i + 1, N):
                if C[i, j] >= tau:
                    H.add_edge(outlets[i], outlets[j], weight=float(C[i, j]))
    return labels(nxc.louvain_communities(H, weight="weight", seed=seed0), outlets)


def eta2(lab, s):
    s = pd.Series(s); g = pd.Series(lab)
    return float(sum(len(v) * (v.mean() - s.mean()) ** 2 for _, v in s.groupby(g)) / ((s - s.mean()) ** 2).sum())


def multilayer(WA, WB, outlets, omegas, n_seeds):
    import igraph as ig, leidenalg as la
    rows, flex = [], []
    graphs = []
    for W in (WA, WB):
        G = centred_graph(W, outlets)
        g = ig.Graph()
        g.add_vertices(len(outlets))
        g.vs["id"] = outlets
        ed = [(outlets.index(a), outlets.index(b)) for a, b in G.edges()]
        g.add_edges(ed)
        w = np.array([G[a][b]["weight"] for a, b in G.edges()])
        g.es["weight"] = list(w / w.mean())            # mean positive tie = 1 in each layer
        graphs.append(g)
    for om in omegas:
        best, switch = None, np.zeros(len(outlets))
        for s in range(n_seeds):
            memb, improv = la.find_partition_temporal(graphs, la.RBConfigurationVertexPartition, interslice_weight=om,
                                                      vertex_id_attr="id", weights="weight", seed=s, n_iterations=-1)
            q = improv
            sw = np.array(memb[0]) != np.array(memb[1])
            switch += sw
            if best is None or q > best[0]:
                best = (q, memb)
        switch /= n_seeds
        q, memb = best
        rows.append(dict(omega=om, quality=q, n_comm_A=len(set(memb[0])), n_comm_B=len(set(memb[1])),
                         n_switch_best=int(np.sum(np.array(memb[0]) != np.array(memb[1]))),
                         switchers_best=", ".join(o for o, a, b in zip(outlets, memb[0], memb[1]) if a != b)))
        flex += [dict(omega=om, outlet=o, p_switch=v, comm_A=a, comm_B=b)
                 for o, v, a, b in zip(outlets, switch, memb[0], memb[1])]
    return pd.DataFrame(rows), pd.DataFrame(flex)


# ---------------------------------------------------------------------------
# per-network statistics (observed and under month-label permutation)
# ---------------------------------------------------------------------------
def outlet_stats(WA, WB, outlets, right):
    N = len(outlets)
    r, drift = np.full(N, np.nan), np.full(N, np.nan)
    R = np.array([o in right for o in outlets])
    for i in range(N):
        m = np.arange(N) != i
        a, b = WA[i, m], WB[i, m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 3:
            r[i] = np.corrcoef(a[ok], b[ok])[0, 1]
        dA = np.nanmean(WA[i, m & R]) - np.nanmean(WA[i, m & ~R])
        dB = np.nanmean(WB[i, m & R]) - np.nanmean(WB[i, m & ~R])
        drift[i] = dB - dA
    return r, drift


def network_stats(WA, WB, outlets, slant, right, n_seeds, seed0):
    labA = consensus(centred_graph(WA, outlets), outlets, n_seeds, seed0)
    labB = consensus(centred_graph(WB, outlets), outlets, n_seeds, seed0 + 7)
    s = np.array([slant[o] for o in outlets])
    bloc = np.array([o in right for o in outlets]).astype(int)
    r, drift = outlet_stats(WA, WB, outlets, right)
    return dict(labA=labA, labB=labB, r=r, drift=drift,
                ARI=adjusted_rand_score(labA, labB), NMI=normalized_mutual_info_score(labA, labB),
                eta2_A=eta2(labA, s), eta2_B=eta2(labB, s),
                NMIbloc_A=normalized_mutual_info_score(bloc, labA), NMIbloc_B=normalized_mutual_info_score(bloc, labB),
                mean_r=float(np.nanmean(r)))


def block_resample(months, block, rng):
    """Moving-block bootstrap of an ordered list of months (same length)."""
    n = len(months)
    starts = rng.integers(0, max(n - block + 1, 1), int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s0, min(s0 + block, n)) for s0 in starts])[:n]
    return [months[i] for i in idx]


def boot_chunk(x, mA, mB, outlets, keys, slant, right, labA, labB, n, n_seeds, block, seed):
    rng = np.random.default_rng(seed)
    xs = {m: g for m, g in x.groupby("month")}
    out = []
    for _ in range(n):
        dA, dB = block_resample(mA, block, rng), block_resample(mB, block, rng)
        WA = tie_matrix(pd.concat([xs[m].assign(month=f"{m}#{k}") for k, m in enumerate(dA)], ignore_index=True),
                        None, outlets, keys)
        WB = tie_matrix(pd.concat([xs[m].assign(month=f"{m}#{k}") for k, m in enumerate(dB)], ignore_index=True),
                        None, outlets, keys)
        st = network_stats(WA, WB, outlets, slant, right, n_seeds, int(rng.integers(1e9)))
        st["ARI_selfA"] = adjusted_rand_score(labA, st["labA"])
        st["ARI_selfB"] = adjusted_rand_score(labB, st["labB"])
        st.pop("labA"); st.pop("labB")
        out.append(st)
    return out


def bh(p):
    p = np.asarray(p, float)
    n = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_change(oc, slant, path, src):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    nm = importlib.import_module("17_network")
    set_style()
    vals = np.array(list(slant.values()))
    norm = TwoSlopeNorm(vcenter=BLOC_CUT[src], vmin=vals.min(), vmax=vals.max())
    panels = [(d, m) for d in ["pol", "nonpol"] for m in ["tfidf", "sbert"]]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 5.2), sharey=True)
    order = sorted(slant, key=lambda o: slant[o])
    for ax, (d, m) in zip(axes, panels):
        g = oc[(oc.domain == d) & (oc.measure == m)].set_index("outlet").reindex(order)
        y = np.arange(len(order))
        ax.axvline(0, color="#777777", lw=0.7)
        lo, hi = g.drift_lo.values, g.drift_hi.values
        ax.hlines(y, lo, hi, color="#d9d9d9", lw=3, zorder=1)
        sig = g.drift_q.values < 0.05
        for yi, v, o, sg in zip(y, g.drift_right.values, order, sig):
            ax.scatter(v, yi, s=34, color=nm.SLANT_CMAP(norm(slant[o])), edgecolors="#222222",
                       linewidths=1.6 if sg else 0.5, zorder=3)
        ax.set_title(f"{DLAB[d]}\n{MLAB[m]}", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([hm.NAMES[o] for o in order], fontsize=6.5)
    fig.text(0.55, 0.005, "Drift towards the right bloc, start → end (grey = 95% block-bootstrap CI; "
             "thick outline = moved more than the median outlet, BH q < .05)", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200); plt.close(fig)


def plot_alluvial(comm, slant, path, src, label_a, label_b):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    nm = importlib.import_module("17_network")
    set_style()
    vals = np.array(list(slant.values()))
    norm = TwoSlopeNorm(vcenter=BLOC_CUT[src], vmin=vals.min(), vmax=vals.max())
    panels = [(d, m) for d in ["pol", "nonpol"] for m in ["tfidf", "sbert"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 9.0))
    for ax, (d, m) in zip(axes.ravel(), panels):
        g = comm[(comm.domain == d) & (comm.measure == m)].copy()
        pos = {}
        for side, col in [(0, "comm_A"), (1, "comm_B")]:
            # order communities by mean slant, outlets by slant within community
            cm = g.groupby(col).slant.mean().sort_values()
            yy, y = {}, 0
            for c in cm.index:
                for o in g[g[col] == c].sort_values("slant").outlet:
                    yy[o] = y; y += 1
                y += 0.8
            pos[side] = yy
            for c in cm.index:
                ys = [yy[o] for o in g[g[col] == c].outlet]
                ax.add_patch(plt.Rectangle((side - 0.06, min(ys) - 0.4), 0.12, max(ys) - min(ys) + 0.8,
                                           fc="#eeeeee", ec="#999999", lw=0.6, zorder=1))
        for o in g.outlet:
            c = nm.SLANT_CMAP(norm(slant[o]))
            moved = g.set_index("outlet").loc[o, "moved"]
            ax.plot([0.06, 0.94], [pos[0][o], pos[1][o]], color=c, lw=2.0 if moved else 0.9,
                    alpha=0.95 if moved else 0.55, zorder=2)
            ax.text(-0.09, pos[0][o], hm.NAMES[o], ha="right", va="center", fontsize=5.6)
            ax.text(1.09, pos[1][o], hm.NAMES[o], ha="left", va="center", fontsize=5.6,
                    fontweight="bold" if moved else "normal")
        ax.set_xlim(-0.6, 1.6)
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.set_title(f"{DLAB[d]}: {MLAB[m]}", fontsize=8)
        ax.text(0, -1.4, label_a, ha="center", fontsize=6.5); ax.text(1, -1.4, label_b, ha="center", fontsize=6.5)
    fig.text(0.5, 0.01, "Consensus Louvain communities (boxes) in each period; line colour = slant; "
             "thick lines / bold names = outlet changed community relative to most of its start-period group",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period-a", default="2014-11:2017-10")
    ap.add_argument("--period-b", default="2021-01:2023-01")
    ap.add_argument("--source", default="pew", choices=["pew", "adfontes"])
    ap.add_argument("--n-boot", type=int, default=999, help="block-bootstrap replicates")
    ap.add_argument("--block", type=int, default=3, help="months per bootstrap block")
    ap.add_argument("--n-qap", type=int, default=4999, help="outlet permutations for QAP / MRQAP")
    ap.add_argument("--n-seeds", type=int, default=200, help="Louvain runs for observed consensus partitions")
    ap.add_argument("--n-seeds-boot", type=int, default=30, help="Louvain runs per consensus inside replicates")
    ap.add_argument("--omegas", default="0.01,0.05,0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--n-seeds-ml", type=int, default=100, help="leidenalg runs per omega")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    if args.quick:
        args.n_boot, args.n_qap, args.n_seeds, args.n_seeds_boot, args.n_seeds_ml = 49, 199, 50, 10, 20
    tag = "_quick" if args.quick else ""
    if args.source != "pew":
        tag += "_" + args.source
    rng = np.random.default_rng(args.seed)

    p = ri.base_panel(ri.AVG_V3)
    slant = ri.slant_dict(args.source)
    cut = BLOC_CUT[args.source]
    right = {o for o, v in slant.items() if v > cut}
    outlets = sorted(slant)
    keys = p.groupby("pair")[["o1", "o2"]].first()
    a0, a1 = args.period_a.split(":"); b0, b1 = args.period_b.split(":")
    months = sorted(p.month.unique())
    mA = [m for m in months if a0 <= m <= a1]
    mB = [m for m in months if b0 <= m <= b1]
    D = np.abs(np.subtract.outer([slant[o] for o in outlets], [slant[o] for o in outlets]))
    isR = np.array([o in right for o in outlets])
    RR = (isR[:, None] & isR[None, :]).astype(float)
    LR = (isR[:, None] ^ isR[None, :]).astype(float)
    typ = np.array([ot.TYPE[o] for o in outlets])
    ST = (typ[:, None] == typ[None, :]).astype(float)
    print(f"Period A {a0}..{a1} ({len(mA)} months), B {b0}..{b1} ({len(mB)} months); source {args.source}", flush=True)

    qap_rows, mrqap_rows, out_rows, comm_rows, glob_rows, ml_rows, flex_rows = [], [], [], [], [], [], []
    for dom in ["pol", "nonpol", "all"]:
        for m in ["tfidf", "sbert"]:
            t0 = time.time()
            x = adjusted_scores(p, dom, m)
            WA, WB = tie_matrix(x, mA, outlets, keys), tie_matrix(x, mB, outlets, keys)
            # 1. QAP persistence
            r, pr = qap_cor(WA, WB, args.n_qap, rng)
            qap_rows.append(dict(domain=dom, measure=m, r_AB=r, p_qap=pr))
            # 2. MRQAP on change
            mq = mrqap_dsp(WB - WA, [D, RR, LR, ST], ["ideological distance D", "both right (RR)",
                                                      "one right (LR)", "same outlet type"], args.n_qap, rng)
            mrqap_rows.append(mq.assign(domain=dom, measure=m, outcome="tie change (B - A)"))
            mq2 = mrqap_dsp(WB, [WA, D, RR, LR, ST], ["tie in A", "ideological distance D", "both right (RR)",
                                                      "one right (LR)", "same outlet type"], args.n_qap, rng)
            mrqap_rows.append(mq2.assign(domain=dom, measure=m, outcome="tie in B"))
            # 3-4. observed stats + block bootstrap within periods
            obs = network_stats(WA, WB, outlets, slant, right, args.n_seeds, args.seed)
            nch = max(1, args.jobs if args.jobs > 0 else os.cpu_count() or 1)
            sizes = [args.n_boot // nch + (1 if i < args.n_boot % nch else 0) for i in range(nch)]
            seeds = rng.integers(0, 2**31 - 1, len(sizes))
            chunks = Parallel(n_jobs=args.jobs)(delayed(boot_chunk)(x, mA, mB, outlets, keys, slant, right,
                                                                    obs["labA"], obs["labB"], s_, args.n_seeds_boot,
                                                                    args.block, int(sd))
                                                for s_, sd in zip(sizes, seeds) if s_ > 0)
            bs = [st for ch in chunks for st in ch]
            Rb = np.array([st["r"] for st in bs]); Db = np.array([st["drift"] for st in bs])
            rel_r_obs = obs["r"] - np.nanmedian(obs["r"])
            rel_d_obs = obs["drift"] - np.nanmedian(obs["drift"])
            rel_r_b = Rb - np.nanmedian(Rb, axis=1, keepdims=True)
            rel_d_b = Db - np.nanmedian(Db, axis=1, keepdims=True)
            two_sided = lambda v: float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)) + 1 / (1 + len(v))))
            p_r = [two_sided(rel_r_b[:, i]) for i in range(len(outlets))]
            p_d = [two_sided(rel_d_b[:, i]) for i in range(len(outlets))]
            q_r, q_d = bh(p_r), bh(p_d)
            for i, o in enumerate(outlets):
                out_rows.append(dict(domain=dom, measure=m, outlet=o, slant=slant[o], right_bloc=o in right,
                                     profile_r=obs["r"][i], profile_r_lo=np.nanpercentile(Rb[:, i], 2.5),
                                     profile_r_hi=np.nanpercentile(Rb[:, i], 97.5),
                                     profile_r_minus_median=rel_r_obs[i], profile_p=p_r[i], profile_q=q_r[i],
                                     drift_right=obs["drift"][i], drift_lo=np.nanpercentile(Db[:, i], 2.5),
                                     drift_hi=np.nanpercentile(Db[:, i], 97.5),
                                     drift_minus_median=rel_d_obs[i], drift_p=p_d[i], drift_q=q_d[i]))
            g = dict(domain=dom, measure=m, n_boot=len(bs))
            ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k in ["ARI", "NMI", "mean_r"]:
                g[k] = obs[k]; g[k + "_lo"], g[k + "_hi"] = ci([st[k] for st in bs])
            g["ARI_selfA_median"] = float(np.nanmedian([st["ARI_selfA"] for st in bs]))
            g["ARI_selfB_median"] = float(np.nanmedian([st["ARI_selfB"] for st in bs]))
            for k in ["eta2", "NMIbloc"]:
                g[k + "_A"], g[k + "_B"] = obs[k + "_A"], obs[k + "_B"]
                v = np.array([st[k + "_B"] - st[k + "_A"] for st in bs])
                g[k + "_change"] = obs[k + "_B"] - obs[k + "_A"]
                g[k + "_change_lo"], g[k + "_change_hi"] = ci(v)
                g[k + "_change_p"] = two_sided(v)
            g["n_comm_A"], g["n_comm_B"] = len(set(obs["labA"])), len(set(obs["labB"]))
            glob_rows.append(g)
            # community membership table; 'moved' = most of the outlet's start-period co-members are not
            # in its end-period community
            for i, o in enumerate(outlets):
                coA = [outlets[j] for j in range(len(outlets)) if j != i and obs["labA"][j] == obs["labA"][i]]
                stay = np.mean([obs["labB"][outlets.index(c)] == obs["labB"][i] for c in coA]) if coA else np.nan
                comm_rows.append(dict(domain=dom, measure=m, outlet=o, slant=slant[o], comm_A=int(obs["labA"][i]),
                                      comm_B=int(obs["labB"][i]), share_startmates_kept=stay,
                                      moved=bool(coA) and stay < 0.5))
            # 5. multilayer
            mlt, flx = multilayer(WA, WB, outlets, [float(w) for w in args.omegas.split(",")], args.n_seeds_ml)
            ml_rows.append(mlt.assign(domain=dom, measure=m)); flex_rows.append(flx.assign(domain=dom, measure=m))
            print(f"  {dom:6s} {m:5s} done in {time.time() - t0:6.1f}s  (r_AB = {r:.2f}, ARI = {obs['ARI']:.2f})",
                  flush=True)

    T = lambda name: os.path.join(TAB, f"netcmp_{name}{tag}.csv")
    pd.DataFrame(qap_rows).to_csv(T("qap"), index=False)
    pd.concat(mrqap_rows).to_csv(T("mrqap"), index=False)
    oc = pd.DataFrame(out_rows); oc.to_csv(T("outlets"), index=False)
    cm = pd.DataFrame(comm_rows); cm.to_csv(T("communities"), index=False)
    gl = pd.DataFrame(glob_rows); gl.to_csv(T("global"), index=False)
    pd.concat(ml_rows).to_csv(T("multilayer"), index=False)
    pd.concat(flex_rows).to_csv(T("multilayer_outlets"), index=False)
    plot_change(oc, slant, os.path.join(FIG, f"fig_network_change{tag}"), args.source)
    plot_alluvial(cm, slant, os.path.join(FIG, f"figS_network_alluvial{tag}"), args.source,
                  f"{a0} – {a1}", f"{b0} – {b1}")

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(qap_rows).round(3).to_string(index=False))
    print(pd.concat(mrqap_rows).round(4).to_string(index=False))
    print(gl.round(3).to_string(index=False))
    sig = oc[(oc.drift_q < .10) | (oc.profile_q < .10)]
    print(sig[["domain", "measure", "outlet", "profile_r", "profile_r_minus_median", "profile_q", "drift_right",
               "drift_lo", "drift_hi", "drift_minus_median", "drift_q"]].round(3).to_string(index=False))
    print(pd.concat(ml_rows)[["domain", "measure", "omega", "n_comm_A", "n_comm_B", "n_switch_best", "switchers_best"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
