"""
common.py -- shared data preparation and estimation utilities.

Everything else in this folder imports from here. No external econometrics
package is required: two-way fixed effects are absorbed by alternating
projections (Frisch-Waugh-Lovell), and the dyadic cluster-robust variance
estimator of Aronow, Samii & Assenova (2015, Political Analysis) is
implemented directly.

Analytic decisions (agreed with the PI after data inspection):
  * v2 (cleaned text, Sep 2026): PRIMARY WINDOW is the full Nov 2014 - Jan 2023.
    The Oct-Nov 2015 TF-IDF break described below was an artefact of URLs and
    stop words in the uncleaned text and is absent from the re-computed scores.
    Starting in Nov 2015 is kept as a robustness check (ALT_START).
  * v1: PRIMARY WINDOW started 2015-11. Every TF-IDF series had a break in
    Oct-Nov 2015 (mean ~0.97 with cross-pair SD ~0.005 before, ~0.87 with
    SD ~0.06 after, in all outlets simultaneously; SBERT shows no break).
    This looks like a pipeline artefact, so the primary window starts once
    the new regime is in place. The full window (2014-11 on) is a robustness check.
  * BLOCS: slant is on Pew's 1-5 audience-ideology scale and is not centred
    at 0. Primary: right = slant > 3 (scale midpoint; NYPost, DCExaminer,
    Fox, DailyCaller, Breitbart). Robustness: median split (<= 2.62 = left).
  * D is mean-centred in all regressions. This changes nothing about delta
    (the D x t coefficient) but makes the month effects / linear trend refer
    to a pair of average ideological distance rather than to D = 0.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                 # project folder (code/ sits directly inside)
# Project layout:
#   auxiliary/   media_slant.csv, media_slant2.csv, sampling_audit.csv
#   samples/     "samples data centred", "samples data uncentred"
#   code/        this folder
#   results/     everything the pipeline writes (results/uncentred/ for the trend stage)
# Which folder of sampled similarity files to read. Default: centred SBERT.
# run_pipeline.py points this at "samples/samples data uncentred" for the trend stage.
SAMPLES_DIR = os.environ.get("SAMPLES_DIR") or os.path.join(ROOT, "samples", "samples data centred")
if not os.path.isabs(SAMPLES_DIR):
    SAMPLES_DIR = os.path.join(ROOT, SAMPLES_DIR)


def sample_path(k):
    """Path of the k-th sampled similarity file (1-5)."""
    return os.path.join(SAMPLES_DIR, f"sample{k}_monthly_pairwise_similarity.csv")


# Which similarity file the pipeline reads. Default: sample 1 (the original
# analysis). Override with the SIM_FILE environment variable, or at run time
# by assigning common.DATA_SIM (see 06_multisample.py).
DATA_SIM = os.environ.get("SIM_FILE", sample_path(1))
DATA_SLANT = os.path.join(ROOT, "auxiliary", "media_slant.csv")     # Pew 2022 audience slant (1-5)
DATA_SLANT2 = os.path.join(ROOT, "auxiliary", "media_slant2.csv")   # Ad Fontes content-based bias score
# Which slant measure the pipeline uses: "pew" (default) or "adfontes".
# Set with the SLANT_SOURCE environment variable or by assigning common.SLANT_SOURCE.
SLANT_SOURCE = os.environ.get("SLANT_SOURCE", "pew")
# Ad Fontes names -> Twitter account names used in the similarity files
ADFONTES_NAMES = {
    "ABC News": "ABC", "BBC News": "BBCWorld", "Breitbart": "BreitbartNews", "Business Insider": "BusinessInsider",
    "BuzzFeed": "BuzzFeedNews", "CBS News": "CBSNews", "CNN": "CNN", "The Daily Caller": "DailyCaller",
    "Fox News": "FoxNews", "The Guardian": "guardian", "HuffPost": "HuffPost", "MSNBC": "MSNBC",
    "NBC News": "NBCNews", "New York Post": "nypost", "The New York Times": "nytimes", "Newsweek": "Newsweek",
    "NPR": "NPR", "POLITICO": "politico", "The Hill": "thehill", "TIME": "TIME", "USA TODAY": "USATODAY",
    "VICE": "VICE", "The Wall Street Journal": "WSJ", "Washington Examiner": "dcexaminer",
    "Washington Post": "washingtonpost",
}
# Bloc cut-point per source: Pew's 1-5 scale is not centred (use its midpoint, 3);
# Ad Fontes is centred at 0 (negative = left), so use the sign.
BLOC_CUT = {"pew": 3.0, "adfontes": 0.0}


def load_slant_frame():
    """Slant table with columns Media (account name) and slant, for the active source."""
    if SLANT_SOURCE == "pew":
        return pd.read_csv(DATA_SLANT)
    a = pd.read_csv(DATA_SLANT2)
    a = a[a.media.isin(ADFONTES_NAMES)]                       # drops PBS, Vox, Limbaugh, Hannity
    return pd.DataFrame({"Media": a.media.map(ADFONTES_NAMES).values, "slant": a.ideo.values})
# Output folder (tables/, figures/, multisample/). Default: results/ in the project folder.
# run_pipeline.py sets OUT_DIR=results/uncentred for the trend stage.
OUT = os.path.abspath(os.environ.get("OUT_DIR") or os.path.join(ROOT, "results"))
# Averaged scores with UNCENTRED SBERT (needed for every common-trend / H1 estimate, since
# centring removes the common trend). Written by the trend stage of run_pipeline.py.
AVG_UNCENTRED = os.environ.get("AVG_UNCENTRED") or os.path.join(
    ROOT, "results", "uncentred", "multisample", "sample_avg_monthly_pairwise_similarity.csv")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
for p in (TAB, FIG):
    os.makedirs(p, exist_ok=True)

PRIMARY_START = "2014-11"   # v2: full window (the 2015 TF-IDF break disappeared after text cleaning)
ALT_START = "2015-11"       # v1 primary window, kept as a robustness check
BREAK_280 = "2017-11"          # 280-character limit
MUSK_LAST = "2022-10"          # last month before ownership change
DOMAINS = {"pol": "Pol", "nonpol": "NonPol", "all": "All"}
DOMAIN_LABEL = {"pol": "Political", "nonpol": "Non-political", "all": "All posts"}
MEASURES = ["tfidf", "sbert", "gap"]
MEASURE_LABEL = {"tfidf": "TF-IDF", "sbert": "SBERT", "gap": "Gap (zTF-IDF - zSBERT)"}

# Okabe-Ito colour-blind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000", "grey": "#999999"}
MEASURE_COL = {"tfidf": OI["blue"], "sbert": OI["orange"], "gap": OI["green"]}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_raw():
    d = pd.read_csv(DATA_SIM)
    s = load_slant_frame()
    return d, s


def build_panel(start=PRIMARY_START, end=None, bloc_rule="midpoint"):
    """Load, restrict to [start, end], and reshape to a pair x month x domain panel.

    Columns: month, o1, o2, pair, domain, tfidf, sbert, gap, t (years since
    Jan 2014), D (|s_i - s_j|), M (midpoint), R (rank distance), bloc flags.
    """
    d, _ = load_raw()
    d = d.rename(columns={"Month": "month", "Outlet 1": "o1", "Outlet 2": "o2"})
    if start:
        d = d[d.month >= start]
    if end:
        d = d[d.month <= end]
    return panel_from_frame(d, bloc_rule=bloc_rule)


def interpolate_panel(start=PRIMARY_START, end=None):
    """Robustness (a): linearly interpolate interior gaps within each pair series.

    Gaps are whole outlet-months, so every pair involving the absent outlet is
    missing. Interpolation is done on the raw six scores (before G is built).
    Leading/trailing gaps are not extrapolated.
    """
    d, s = load_raw()
    d = d.rename(columns={"Month": "month", "Outlet 1": "o1", "Outlet 2": "o2"})
    if start:
        d = d[d.month >= start]
    if end:
        d = d[d.month <= end]
    months = sorted(d.month.unique())
    cols = [c for c in d.columns if c.startswith(("TF-IDF", "SBERT"))]
    out = []
    for (a, b), g in d.groupby(["o1", "o2"]):
        g = g.set_index("month").reindex(months)
        g[cols] = g[cols].interpolate(method="linear", limit_area="inside")
        g["o1"], g["o2"] = a, b
        g["Sample"] = 1
        out.append(g.dropna(subset=cols).reset_index())
    di = pd.concat(out, ignore_index=True).rename(columns={"index": "month"})
    return di


def panel_from_frame(d, bloc_rule="midpoint"):
    """Reshape a raw frame (cols renamed to month/o1/o2) into the long panel.

    The gap measure G = z(TF-IDF) - z(SBERT) is standardised within domain on
    whatever sample is passed in.
    """
    s = load_slant_frame()
    slant = dict(zip(s.Media, s.slant))
    rank = dict(zip(s.Media, s.slant.rank(method="average")))
    rows = []
    for dom, suf in DOMAINS.items():
        x = d[["month", "o1", "o2"]].copy()
        x["domain"] = dom
        x["tfidf"] = d[f"TF-IDF-{suf}"].values
        x["sbert"] = d[f"SBERT-{suf}"].values
        rows.append(x)
    p = pd.concat(rows, ignore_index=True)
    p["pair"] = p.o1 + "|" + p.o2
    per = pd.PeriodIndex(p.month, freq="M")
    p["t"] = ((per.year - 2014) * 12 + (per.month - 1)) / 12.0
    p["year"] = per.year
    s1, s2 = p.o1.map(slant), p.o2.map(slant)
    p["s1"], p["s2"] = s1, s2
    p["D"] = (s1 - s2).abs()
    p["M"] = (s1 + s2) / 2
    p["R"] = (p.o1.map(rank) - p.o2.map(rank)).abs()
    cut = BLOC_CUT[SLANT_SOURCE] if bloc_rule == "midpoint" else float(np.median(list(slant.values())))
    p["right1"] = (s1 > cut).astype(int)
    p["right2"] = (s2 > cut).astype(int)
    p["within"] = (p.right1 == p.right2).astype(int)
    p["bloc_type"] = np.select([(p.right1 + p.right2) == 0, (p.right1 + p.right2) == 2],
                               ["LL", "RR"], "LR")
    for dom in DOMAINS:
        m = p.domain == dom
        zt = (p.loc[m, "tfidf"] - p.loc[m, "tfidf"].mean()) / p.loc[m, "tfidf"].std()
        zs = (p.loc[m, "sbert"] - p.loc[m, "sbert"].mean()) / p.loc[m, "sbert"].std()
        p.loc[m, "gap"] = zt - zs
    return p.reset_index(drop=True), slant


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
FALLBACK_LOG = []   # models where the dyadic VCOV was non-PSD (see vcov_dyadic)


def _codes(v):
    return pd.factorize(pd.Series(v).astype(str))[0]


def demean(M, fe_list, tol=1e-12, maxit=10000):
    """Partial out several sets of fixed effects by alternating projections.

    M: (n, k) float array. fe_list: list of integer code arrays (length n).
    Works for unbalanced panels.
    """
    M = np.array(M, dtype=float, copy=True)
    if M.ndim == 1:
        M = M[:, None]
    counts = [np.bincount(g) for g in fe_list]
    for _ in range(maxit):
        Mold = M.copy()
        for g, c in zip(fe_list, counts):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(g, weights=M[:, j], minlength=len(c)) / c)[g]
        if np.max(np.abs(M - Mold)) < tol:
            break
    return M


def vcov_dyadic(Xt, e, o1, o2, dyad, XtX_inv):
    """Aronow-Samii-Assenova (2015) dyadic cluster-robust covariance.

    Scores are summed within dyad over time (so arbitrary serial correlation
    within a dyad is allowed) and then any two dyads that share an outlet
    are allowed to be correlated. meat = sum_i U_i U_i' - sum_d u_d u_d',
    where u_d is the dyad score and U_i the sum of scores of every dyad
    involving outlet i. The subtraction removes the double-counted d = d'
    terms. Small-sample factor G/(G-1), G = number of outlets.
    """
    k = Xt.shape[1]
    S = Xt * e[:, None]
    dcode = _codes(dyad)
    nd = dcode.max() + 1
    u = np.column_stack([np.bincount(dcode, weights=S[:, j], minlength=nd) for j in range(k)])
    # map each dyad to its two outlets
    first = pd.DataFrame({"d": dcode, "a": o1, "b": o2}).drop_duplicates("d").sort_values("d")
    outlets = pd.Index(sorted(set(first.a) | set(first.b)))
    ia, ib = outlets.get_indexer(first.a), outlets.get_indexer(first.b)
    G = len(outlets)
    U = np.zeros((G, k))
    np.add.at(U, ia, u)
    np.add.at(U, ib, u)
    meat = U.T @ U - u.T @ u
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    # The ASA estimator is not guaranteed positive semi-definite. Two fixes:
    #  (1) if every variance on the diagonal is positive, clip negative
    #      eigenvalues to zero (Cameron, Gelbach & Miller 2011, the standard
    #      multiway-clustering fix);
    #  (2) if some variance is itself <= 0 (seen for a between-bloc indicator
    #      that is almost perfectly balanced within every outlet), use the
    #      conservative PSD version that counts each dyad's own term twice
    #      (meat = sum_i U_i U_i'), which is 5-35% larger in our models.
    # Each case is recorded in FALLBACK_LOG.
    fix = None
    Vs = (V + V.T) / 2
    w, Q = np.linalg.eigh(Vs)
    if w.min() < 0:
        if np.all(np.diag(Vs) > 0):
            V, fix = Q @ np.diag(np.clip(w, 0, None)) @ Q.T, "eigen-clip"
        else:
            V, fix = XtX_inv @ (U.T @ U) @ XtX_inv * G / (G - 1), "conservative"
    vcov_dyadic.last_fallback = fix
    return V, G


def vcov_cluster(Xt, e, cl, XtX_inv):
    k = Xt.shape[1]
    S = Xt * e[:, None]
    c = _codes(cl)
    G = c.max() + 1
    u = np.column_stack([np.bincount(c, weights=S[:, j], minlength=G) for j in range(k)])
    return XtX_inv @ (u.T @ u) @ XtX_inv * G / (G - 1), G


def vcov_twoway(Xt, e, o1, o2, dyad, XtX_inv):
    """Cameron-Gelbach-Miller two-way clustering on outlet-1 and outlet-2.

    Note this depends on which outlet is stored as 'Outlet 1' (here the
    order of the slant file) and ignores correlation between (i,j) and
    (k,i) -- a known weakness relative to the dyadic estimator.
    """
    V1, G1 = vcov_cluster(Xt, e, o1, XtX_inv)
    V2, G2 = vcov_cluster(Xt, e, o2, XtX_inv)
    V12, _ = vcov_cluster(Xt, e, pd.Series(o1).astype(str) + "|" + pd.Series(o2).astype(str), XtX_inv)
    return V1 + V2 - V12, min(G1, G2)


def fe_ols(df, y, xvars, fe, o1="o1", o2="o2", dyad="pair", cluster=None):
    """OLS with absorbed fixed effects and dyadic + two-way clustered SEs.

    df: DataFrame (rows with NA in y/x are dropped). fe: list of column names
    (each a categorical FE). Returns dict with coef table and metadata.
    If `cluster` is given (a column), a one-way cluster SE replaces the dyadic
    one (used for outlet-level models).
    """
    df = df.dropna(subset=[y] + xvars).reset_index(drop=True)
    fe_codes = [_codes(df[f]) for f in fe]
    Z = demean(np.column_stack([df[y].values] + [df[x].values for x in xvars]), fe_codes)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.pinv(Xt.T @ Xt)
    b = XtX_inv @ Xt.T @ yt
    e = yt - Xt @ b
    res = {"n": len(df), "y": y, "x": xvars}
    if cluster is None:
        Vd, G = vcov_dyadic(Xt, e, df[o1].values, df[o2].values, df[dyad].values, XtX_inv)
        Vt, Gt = vcov_twoway(Xt, e, df[o1].values, df[o2].values, df[dyad].values, XtX_inv)
        res.update(n_pairs=df[dyad].nunique(), n_months=df["month"].nunique(), G=G,
                   dyadic_fallback=vcov_dyadic.last_fallback)
        if vcov_dyadic.last_fallback:
            FALLBACK_LOG.append((vcov_dyadic.last_fallback, y, tuple(xvars), len(df)))
        se_d, se_t = np.sqrt(np.clip(np.diag(Vd), 0, None)), np.sqrt(np.clip(np.diag(Vt), 0, None))
        dfree = G - 1
        tab = pd.DataFrame({"coef": b, "se_dyadic": se_d, "se_twoway": se_t}, index=xvars)
        tab["t_dyadic"] = tab.coef / tab.se_dyadic
        tab["p_dyadic"] = 2 * stats.t.sf(np.abs(tab.t_dyadic), dfree)
        tab["p_twoway"] = 2 * stats.t.sf(np.abs(tab.coef / tab.se_twoway), Gt - 1)
        tab["lo"] = tab.coef - stats.t.ppf(.975, dfree) * tab.se_dyadic
        tab["hi"] = tab.coef + stats.t.ppf(.975, dfree) * tab.se_dyadic
        res["V"] = Vd
    else:
        Vc, G = vcov_cluster(Xt, e, df[cluster].values, XtX_inv)
        se = np.sqrt(np.diag(Vc))
        tab = pd.DataFrame({"coef": b, "se": se}, index=xvars)
        tab["p"] = 2 * stats.t.sf(np.abs(b / se), G - 1)
        tab["lo"] = b - stats.t.ppf(.975, G - 1) * se
        tab["hi"] = b + stats.t.ppf(.975, G - 1) * se
        res.update(G=G, n_units=df[cluster].nunique(), n_months=df["month"].nunique())
        res["V"] = Vc
    # within R2
    res["r2_within"] = 1 - (e @ e) / (yt @ yt)
    res["tab"] = tab
    res["resid"] = e
    res["data"] = df
    return res


def lincom(res, w):
    """Linear combination w'b with dyadic SE (w: dict var->weight)."""
    tab, V = res["tab"], res["V"]
    wv = np.array([w.get(v, 0.0) for v in tab.index])
    est = float(wv @ tab.coef.values)
    se = float(np.sqrt(wv @ V @ wv))
    p = 2 * stats.t.sf(abs(est / se), res["G"] - 1)
    return est, se, p


def month_effects(df, y, xvars, pair="pair", month="month"):
    """Recover month fixed effects gamma_t from Sim = a_ij + g_t + X b.

    After estimating b by FWL, y - Xb is regressed on pair + month dummies
    by alternating projections; the month means of the pair-demeaned
    remainder give gamma_t (normalised so the first month = 0).
    """
    r = fe_ols(df, y, xvars, [pair, month])
    d = r["data"]
    ytil = d[y].values - d[xvars].values @ r["tab"].coef.values
    pc, mc = _codes(d[pair]), _codes(d[month])
    # solve for both FE sets by alternating means (Gauss-Seidel)
    a = np.zeros(pc.max() + 1)
    g = np.zeros(mc.max() + 1)
    for _ in range(5000):
        g_old = g.copy()
        a = np.bincount(pc, weights=ytil - g[mc]) / np.bincount(pc)
        g = np.bincount(mc, weights=ytil - a[pc]) / np.bincount(mc)
        if np.max(np.abs(g - g_old)) < 1e-12:
            break
    months = pd.Series(pd.factorize(d[month].astype(str))[1])
    out = pd.DataFrame({"month": months, "gamma": g}).sort_values("month").reset_index(drop=True)
    out["gamma"] -= out.gamma.iloc[0]
    return out


def save_fallback_log(name):
    """Write which models needed a non-PSD fix of the dyadic VCOV."""
    pd.DataFrame(FALLBACK_LOG, columns=["fix", "outcome", "regressors", "n"]).to_csv(
        os.path.join(TAB, f"vcov_fixes_{name}.csv"), index=False)
    FALLBACK_LOG.clear()


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "+" if p < .10 else ""


def fmt(b, se, p, digits=4):
    return f"{b:.{digits}f}{stars(p)}", f"({se:.{digits}f})"


# Figure titles are off by default: APA puts titles in the caption, not the image.
# Set FIG_TITLES=1 in the environment to embed them (e.g. for slides).
SHOW_TITLES = os.environ.get("FIG_TITLES", "0") == "1"


def suptitle(fig, *args, **kwargs):
    if SHOW_TITLES:
        fig.suptitle(*args, **kwargs)


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
