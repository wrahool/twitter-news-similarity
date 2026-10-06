"""
18_robust_inference.py -- inference that is robust to network dependence.

Why: the dyadic cluster-robust SEs used throughout (Aronow, Samii & Assenova
2015) allow any two pair-months that share an outlet to be correlated, but
(i) treat pairs with no outlet in common (e.g. Fox-Breitbart, DailyCaller-NYPost)
as independent given the fixed effects, (ii) rely on asymptotics in the number
of outlets (G = 25), and (iii) for the common-trend models (H1, pair FE + beta*t)
ignore shocks common to all pairs in a month, which are not absorbed when the
model has no month fixed effects. This script re-tests the key estimates with:

  dyadic      the analytic estimator used so far (for reference)
  DK          Driscoll-Kraay SEs (TREND TERMS ONLY): scores summed over all pairs
              within a month, then Newey-West across months (Bartlett, --dk-lag
              months). Robust to any cross-sectional dependence, incl. month-wide
              shocks. Not used for slant terms: with month FE their identifying
              variation is persistent pair-level (D_ij x t), and pair-level serial
              correlation beyond the lag makes DK anti-conservative there.
  dyadic+time two-way combination for trend terms (Cameron-Gelbach-Miller style):
              V_dyadic + V_DK - V_pair-month. The most conservative analytic
              choice for H1 beta.
  perm        outlet-permutation test: slant scores are shuffled across the 25
              outlets (blocs, distance, midpoint and rank are rebuilt from the
              shuffled scores) and the model re-estimated. Valid under arbitrary
              network dependence among outlets, assuming outlets are exchangeable
              under the null. Reported for the raw coefficient and for the
              studentised statistic (coef / dyadic SE). Only for slant terms.
  boot        pigeonhole (node) bootstrap for dyadic data (Davezies, D'Haultfoeuille
              & Guyonvarch 2021): resample the 25 outlets with replacement and keep
              every pair of distinct draws; SE = SD of the bootstrap estimates,
              plus a percentile 95% CI. Known to be somewhat conservative.
  jackknife   delete-one-outlet jackknife SE (t with G - 1 df).

Data: scores averaged over the five samples. Trend terms (H1) use the UNCENTRED
SBERT (results/uncentred, which centring would make trendless); everything else uses
centred SBERT (results/). TF-IDF is identical in both. Count-control variants
add the post-count terms from 15_postcount_check.py (political / non-political).

Usage (headless):
    python 18_robust_inference.py                     # full run (defaults below)
    python 18_robust_inference.py --quick             # 99 perms / 49 boots, smoke test
    python 18_robust_inference.py --jobs 16 --n-perm 4999 --n-boot 1999
Outputs: tables/robust_inference.csv, tables/robust_inference_draws.csv.gz,
         figures/figS_robust_inference.(pdf|png)
"""
import os, sys, time, argparse, importlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per worker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *
from joblib import Parallel, delayed

pc = importlib.import_module("15_postcount_check")
am = importlib.import_module("13_adjusted_means")
ot = importlib.import_module("08_outlet_type")
AVG_V2 = AVG_UNCENTRED                      # uncentred SBERT (see common.py)
AVG_V3 = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")
TYPES = ["broadcast", "newspaper", "magazine", "digital", "uk"]
CTRL = pc.CTRL


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def slant_dict(src):
    common.SLANT_SOURCE = src
    s = load_slant_frame()
    return dict(zip(s.Media, s.slant))


def base_panel(path):
    """Long panel (pair x month x domain) with post counts; slant-free columns only."""
    common.DATA_SIM = path
    p, _ = build_panel()
    p = pc.add_counts(p)
    keep = ["month", "o1", "o2", "pair", "domain", "tfidf", "sbert", "t", "year",
            "Nsum", "Nprod", "full"]
    return p[keep].reset_index(drop=True)


def attach(p, slant, cut):
    """(Re)build every slant-derived regressor from an outlet -> slant mapping."""
    p = p.copy()
    s1, s2 = p.o1.map(slant).values, p.o2.map(slant).values
    D, M = np.abs(s1 - s2), (s1 + s2) / 2
    first = p.drop_duplicates("pair")
    Dbar = np.abs(first.o1.map(slant).values - first.o2.map(slant).values).mean()
    Mbar = ((first.o1.map(slant).values + first.o2.map(slant).values) / 2).mean()
    r1, r2 = (s1 > cut).astype(int), (s2 > cut).astype(int)
    p["D"], p["M"] = D, M
    p["Dt"] = (D - Dbar) * p.t.values
    p["Mt"] = (M - Mbar) * p.t.values
    p["RRt"] = ((r1 + r2) == 2) * p.t.values
    p["LRt"] = ((r1 + r2) == 1) * p.t.values
    return p


def pair_level_frame(p, measure, domain, controls):
    """Month-adjusted pair means (pair FE from y = a_pair + g_month [+ count terms])."""
    x = p[p.domain == domain]
    y = x[measure].values
    if controls:
        g = fe_ols(x, measure, CTRL, ["pair", "month"])["tab"].coef
        y = y - sum(g[c] * x[c].values for c in CTRL)
    a = am.pair_effects(x.assign(yy=y), "yy")
    k = x.groupby("pair")[["o1", "o2"]].first()
    k["y"] = a.reindex(k.index).values
    k["month"] = "all"
    return k.reset_index()


# ---------------------------------------------------------------------------
# estimators: each returns (coef, dyadic_se, dk_se, dyadic+time_se) for the target term
# ---------------------------------------------------------------------------
def dk_vcov(Xt, e, months, XtX_inv, lag):
    """Driscoll-Kraay: month-summed scores, Newey-West (Bartlett) over months."""
    S = Xt * e[:, None]
    mc, order = pd.factorize(pd.Series(months))
    srt = np.argsort(order.values)                      # chronological
    k = Xt.shape[1]
    H = np.column_stack([np.bincount(mc, weights=S[:, j], minlength=len(order)) for j in range(k)])[srt]
    T = H.shape[0]
    meat = H.T @ H
    for L in range(1, min(lag, T - 1) + 1):
        w = 1 - L / (lag + 1)
        G_ = H[L:].T @ H[:-L]
        meat += w * (G_ + G_.T)
    return XtX_inv @ meat @ XtX_inv * T / (T - 1)


def ols_fe(df, y, xvars, fe, target, lag, need_dk=True, dk_col="month"):
    df = df.dropna(subset=[y] + xvars).reset_index(drop=True)
    codes = [pd.factorize(df[f].astype(str))[0] for f in fe]
    Z = demean(np.column_stack([df[y].values] + [df[v].values for v in xvars]), codes)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.pinv(Xt.T @ Xt)
    b = XtX_inv @ Xt.T @ yt
    e = yt - Xt @ b
    j = xvars.index(target)
    V, _ = vcov_dyadic(Xt, e, df.o1.values, df.o2.values, df.pair.values, XtX_inv)
    se_d = float(np.sqrt(max(V[j, j], 0.0)))
    if not need_dk:
        return float(b[j]), se_d, np.nan, np.nan
    Vdk = dk_vcov(Xt, e, df[dk_col].values, XtX_inv, lag)
    S = Xt * e[:, None]
    Vw = XtX_inv @ (S.T @ S) @ XtX_inv                      # pair-month (intersection) term
    v2 = V[j, j] + Vdk[j, j] - Vw[j, j]                     # two-way: outlet-dyadic + time (CGM-style)
    v2 = v2 if v2 > 0 else max(V[j, j], Vdk[j, j])
    return float(b[j]), se_d, float(np.sqrt(max(Vdk[j, j], 0.0))), float(np.sqrt(v2))


def ols_xsec(df, xvars, target):
    """Cross-section of pair levels: y ~ 1 + xvars, dyadic SE."""
    X = np.column_stack([np.ones(len(df))] + [df[v].values for v in xvars])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ df.y.values
    e = df.y.values - X @ b
    V, _ = vcov_dyadic(X, e, df.o1.values, df.o2.values, df.pair.values, XtX_inv)
    j = 1 + xvars.index(target)
    return float(b[j]), float(np.sqrt(max(V[j, j], 0.0))), np.nan, np.nan


def type_counts(df):
    """Effect-coded outlet-type counts (0/1/2 outlets of each type), reference = last type."""
    out = {}
    n = {k: (df.o1.map(ot.TYPE) == k).astype(float).values + (df.o2.map(ot.TYPE) == k).astype(float).values
         for k in TYPES}
    for k in TYPES[:-1]:
        out["n_" + k] = n[k] - n[TYPES[-1]]
    return out


# ---------------------------------------------------------------------------
# specifications
# ---------------------------------------------------------------------------
def specs():
    """Key estimates. kind: 'trend' (no slant term), 'panel' (pair+month FE), 'h3', 'xsec'."""
    S = []
    for dom in ["nonpol", "pol"]:
        for m in ["tfidf", "sbert"]:
            S.append(dict(name=f"H1 beta, {dom}, {m} (uncentred)", family="H1 trend beta", kind="trend",
                          data="v2", domain=dom, measure=m, target="t", ctrl=False))
            S.append(dict(name=f"H1 beta, {dom}, {m} (uncentred), count controls", family="H1 trend beta",
                          kind="trend", data="v2", domain=dom, measure=m, target="t", ctrl=True))
    for dom in ["pol", "nonpol"]:
        for m in ["tfidf", "sbert"]:
            for xs in [["D"], ["D", "types"]]:
                S.append(dict(name=f"Level gradient, {dom}, {m}" + (" + format/UK" if len(xs) > 1 else ""),
                              family="Level gradient (pair means on D)", kind="xsec", data="v3", domain=dom,
                              measure=m, target="D", xs=xs, ctrl=True))
    for dom in ["pol", "nonpol"]:
        for m in ["tfidf", "sbert"]:
            for ctrl in [False, True]:
                S.append(dict(name=f"H2 delta (D x t), {dom}, {m}" + (", count controls" if ctrl else ""),
                              family="H2 delta", kind="panel", data="v3", domain=dom, measure=m,
                              xvars=["Dt"], target="Dt", ctrl=ctrl))
    for m in ["tfidf", "sbert"]:
        for ctrl in [False, True]:
            S.append(dict(name=f"H3 D x t x Pol, {m}" + (", count controls" if ctrl else ""), family="H3 contrast",
                          kind="h3", data="v3", domain="pol-nonpol", measure=m, target="Dt_pol", ctrl=ctrl))
    for dom in ["all", "nonpol", "pol"]:
        for ctrl in ([False] if dom == "all" else [False, True]):
            S.append(dict(name=f"RR x t (right-right vs left-left), {dom}, sbert" + (", count controls" if ctrl else ""),
                          family="RQ right-right convergence", kind="panel", data="v3", domain=dom, measure="sbert",
                          xvars=["LRt", "RRt"], target="RRt", ctrl=ctrl))
    for dom in ["all", "nonpol"]:
        S.append(dict(name=f"M x t, {dom}, sbert", family="RQ midpoint x t", kind="panel", data="v3", domain=dom,
                      measure="sbert", xvars=["Dt", "Mt"], target="Mt", ctrl=False))
    return S


def estimate(sp, p, slant, cut, lag, need_dk=True):
    """Fit one specification on panel p (slant-free columns) with an outlet->slant map."""
    if sp["kind"] == "xsec":
        lev = sp["_levels"].copy()                         # pair levels do not depend on slant
        s1, s2 = lev.o1.map(slant).values, lev.o2.map(slant).values
        lev["D"] = np.abs(s1 - s2)
        xv = ["D"]
        if "types" in sp["xs"]:
            for k, v in type_counts(lev).items():
                lev[k] = v
            xv += ["n_" + k for k in TYPES[:-1]]
        return ols_xsec(lev, xv, "D")
    ctrl = CTRL if sp["ctrl"] else []
    if sp["kind"] == "trend":
        x = attach(p[p.domain == sp["domain"]], slant, cut)
        return ols_fe(x, sp["measure"], ["t", "Dt"] + ctrl, ["pair"], "t", lag, need_dk)
    if sp["kind"] == "panel":
        x = attach(p[p.domain == sp["domain"]], slant, cut)
        return ols_fe(x, sp["measure"], sp["xvars"] + ctrl, ["pair", "month"], sp["target"], lag, need_dk)
    if sp["kind"] == "h3":
        s = attach(p[p.domain.isin(["pol", "nonpol"])], slant, cut)
        s["pol"] = (s.domain == "pol").astype(int)
        s["Dt_pol"] = s.Dt * s.pol
        s["pair_dom"], s["month_dom"] = s.pair + "#" + s.domain, s.month + "#" + s.domain
        cx = []
        for c in ctrl:
            s[c + "_pol"] = s[c] * s.pol
            cx += [c, c + "_pol"]
        s = s.rename(columns={"pair": "pair_orig"}).rename(columns={"pair_dom": "pair"})
        # clustering still by outlet: dyad id = pair x domain is nested in the outlet pair, which is what
        # vcov_dyadic needs (it maps each dyad id to its two outlets)
        return ols_fe(s, sp["measure"], ["Dt", "Dt_pol"] + cx, ["pair", "month_dom"],
                      "Dt_pol", lag, False)
    raise ValueError(sp["kind"])


# ---------------------------------------------------------------------------
# resampling schemes
# ---------------------------------------------------------------------------
def node_bootstrap_panel(p, draw, outlets):
    """Pigeonhole bootstrap: keep every pair of distinct draws (a < b) with different outlets."""
    rows = []
    by_pair = {k: g for k, g in p.groupby("pair", sort=False)}
    for a in range(len(draw)):
        for b in range(a + 1, len(draw)):
            u, v = outlets[draw[a]], outlets[draw[b]]
            if u == v:
                continue
            key = f"{u}|{v}" if f"{u}|{v}" in by_pair else f"{v}|{u}"
            g = by_pair.get(key)
            if g is None:
                continue
            g = g.copy()
            # copies get their own ids; slant is looked up through the original name
            ua, vb = (u, v) if g.o1.iloc[0] == u else (v, u)
            ia, ib = (a, b) if g.o1.iloc[0] == u else (b, a)
            g["orig1"], g["orig2"] = g.o1, g.o2
            g["o1"], g["o2"] = f"{g.o1.iloc[0]}#{ia}", f"{g.o2.iloc[0]}#{ib}"
            g["pair"] = g.o1 + "|" + g.o2
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def run_spec(sp, src, panels, slants, args, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    slant, cut = slants[src], BLOC_CUT[src]
    p = panels[sp["data"]]
    if sp["kind"] == "xsec":
        sp = dict(sp, _levels=pair_level_frame(p, sp["measure"], sp["domain"], sp["ctrl"]))
    b, se_d, se_dk, se_2w = estimate(sp, p, slant, cut, args.dk_lag, need_dk=(sp["kind"] == "trend"))
    t_obs = b / se_d if se_d > 0 else np.nan
    outlets = sorted(slant)
    G = len(outlets)
    draws = []

    # permutation (slant terms only)
    p_perm_b = p_perm_t = np.nan
    if sp["kind"] != "trend":
        pb, pt = [], []
        vals = np.array([slant[o] for o in outlets])
        for _ in range(args.n_perm):
            sl = dict(zip(outlets, rng.permutation(vals)))
            bb, ss = estimate(sp, p, sl, cut, args.dk_lag, need_dk=False)[:2]
            pb.append(bb); pt.append(bb / ss if ss > 0 else np.nan)
        pb, pt = np.array(pb), np.array(pt)
        p_perm_b = (1 + np.sum(np.abs(pb) >= abs(b))) / (1 + len(pb))
        p_perm_t = (1 + np.sum(np.abs(pt) >= abs(t_obs))) / (1 + np.isfinite(pt).sum())
        draws += [dict(scheme="perm", stat="coef", value=v) for v in pb]

    # jackknife (drop one outlet)
    jk = []
    for o in outlets:
        q = p[(p.o1 != o) & (p.o2 != o)]
        spj = sp
        if sp["kind"] == "xsec":
            lev = sp["_levels"]
            spj = dict(sp, _levels=lev[(lev.o1 != o) & (lev.o2 != o)])
        jk.append(estimate(spj, q, slant, cut, args.dk_lag, need_dk=False)[0])
    jk = np.array(jk)
    se_jk = float(np.sqrt((G - 1) / G * np.sum((jk - jk.mean()) ** 2)))

    # pigeonhole (node) bootstrap
    bs = []
    for _ in range(args.n_boot):
        draw = rng.integers(0, G, G)
        if sp["kind"] == "xsec":
            lev = node_bootstrap_panel(sp["_levels"], draw, outlets)
            bb = estimate_boot_xsec(dict(sp, _levels=lev), slant)
        else:
            q = node_bootstrap_panel(p, draw, outlets)
            sl = {**slant, **{o: slant[o.split("#")[0]] for o in set(q.o1) | set(q.o2)}}
            bb = estimate(sp, q, sl, cut, args.dk_lag, need_dk=False)[0]
        bs.append(bb)
    bs = np.array(bs)
    se_boot = float(np.nanstd(bs, ddof=1))
    draws += [dict(scheme="boot", stat="coef", value=v) for v in bs]
    draws += [dict(scheme="jackknife", stat="coef", value=v, dropped=o) for v, o in zip(jk, outlets)]

    tp = lambda est, se: 2 * stats.t.sf(abs(est / se), G - 1) if se and se > 0 else np.nan
    row = dict(name=sp["name"], family=sp["family"], kind=sp["kind"], data=sp["data"], domain=sp["domain"],
               measure=sp["measure"], count_controls=sp["ctrl"], source=src, coef=b,
               se_dyadic=se_d, p_dyadic=tp(b, se_d),
               se_dk=se_dk, p_dk=(2 * stats.norm.sf(abs(b / se_dk)) if se_dk and se_dk > 0 else np.nan),
               se_dyadic_time=se_2w, p_dyadic_time=tp(b, se_2w),
               p_perm_coef=p_perm_b, p_perm_t=p_perm_t,
               se_boot=se_boot, p_boot=tp(b, se_boot),
               boot_lo=float(np.nanpercentile(bs, 2.5)), boot_hi=float(np.nanpercentile(bs, 97.5)),
               se_jackknife=se_jk, p_jackknife=tp(b, se_jk),
               n_perm=args.n_perm if sp["kind"] != "trend" else 0, n_boot=args.n_boot,
               seconds=round(time.time() - t0, 1))
    for d in draws:
        d.update(name=sp["name"], source=src)
    print(f"  [{row['seconds']:7.1f}s] {src:8s} {sp['name']}", flush=True)
    return row, draws


def estimate_boot_xsec(sp, slant):
    """Cross-section on a bootstrap sample: attributes via original names, clustering via copy ids."""
    lev = sp["_levels"].copy()
    s1, s2 = lev.orig1.map(slant).values, lev.orig2.map(slant).values
    lev["D"] = np.abs(s1 - s2)
    xv = ["D"]
    if "types" in sp["xs"]:
        tmp = lev.assign(o1=lev.orig1, o2=lev.orig2)
        for k, v in type_counts(tmp).items():
            lev[k] = v
        xv += ["n_" + k for k in TYPES[:-1]]
    return ols_xsec(lev, xv, "D")[0]


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def plot(res, path):
    import matplotlib.pyplot as plt
    set_style()
    r = res[res.source == "pew"].reset_index(drop=True)
    r = r.assign(p_dk=r.p_dyadic_time)
    methods = [("p_dyadic", "Dyadic (analytic)", OI["black"], "o"), ("p_dk", "Dyadic + time (trends only)", OI["orange"], "s"),
               ("p_perm_t", "Outlet permutation (studentised)", OI["vermillion"], "D"),
               ("p_boot", "Node bootstrap", OI["blue"], "^"), ("p_jackknife", "Outlet jackknife", OI["green"], "v")]
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(r) + 1.2))
    for k, (col, lab, c, mk) in enumerate(methods):
        y = np.arange(len(r))[::-1] + (k - 2) * 0.12
        ax.scatter(np.clip(r[col], 1e-4, 1), y, s=16, color=c, marker=mk, label=lab, zorder=3)
    ax.axvline(0.05, color="#555555", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.2)
    ax.set_yticks(np.arange(len(r))[::-1])
    ax.set_yticklabels([f"{n}  ({b:+.4f})" for n, b in zip(r.name, r.coef)], fontsize=6.3)
    ax.set_xlabel("p-value (log scale; dashed line = .05). Pew slant.")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.06 * 30 / max(len(r), 1)),
              ncol=3)
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight"); fig.savefig(path + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=1999)
    ap.add_argument("--n-boot", type=int, default=999)
    ap.add_argument("--dk-lag", type=int, default=12, help="Driscoll-Kraay Bartlett lag in months")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--sources", default="pew,adfontes")
    ap.add_argument("--only", default="", help="substring filter on specification names")
    ap.add_argument("--quick", action="store_true", help="99 permutations, 49 bootstraps (smoke test)")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    if args.quick:
        args.n_perm, args.n_boot = 99, 49

    panels = {"v2": base_panel(AVG_V2), "v3": base_panel(AVG_V3)}
    sources = [s for s in args.sources.split(",") if s]
    slants = {s: slant_dict(s) for s in sources}
    S = [s for s in specs() if args.only.lower() in s["name"].lower()]
    tasks = [(sp, src) for sp in S for src in sources]
    print(f"{len(tasks)} estimates x ({args.n_perm} perms + {args.n_boot} boots + 25 jackknife); "
          f"jobs = {args.jobs}", flush=True)
    seeds = np.random.SeedSequence(args.seed).spawn(len(tasks))
    out = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_spec)(sp, src, panels, slants, args, int(sd.generate_state(1)[0])) for (sp, src), sd in zip(tasks, seeds))
    res = pd.DataFrame([o[0] for o in out])
    draws = pd.DataFrame([d for o in out for d in o[1]])
    tag = "_quick" if args.quick else ""
    res.to_csv(os.path.join(TAB, f"robust_inference{tag}.csv"), index=False)
    draws.to_csv(os.path.join(TAB, f"robust_inference_draws{tag}.csv.gz"), index=False)
    plot(res, os.path.join(FIG, f"figS_robust_inference{tag}"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
    cols = ["source", "name", "coef", "p_dyadic", "p_dk", "p_dyadic_time", "p_perm_coef", "p_perm_t", "p_boot", "p_jackknife"]
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
