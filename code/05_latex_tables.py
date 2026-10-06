"""
05_latex_tables.py -- write APA-style LaTeX tables (one file per table) for
the manuscript, reading only the CSVs produced by 01-04.

Output: outputs/manuscript_tables/*.tex  (copied into manuscript/tables/)
Requires in the LaTeX preamble: booktabs, threeparttable, rotating.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUTT = os.path.join(OUT, "manuscript_tables")
os.makedirs(OUTT, exist_ok=True)
DL = {"pol": "Political", "nonpol": "Non-political", "all": "All posts"}
ML = {"tfidf": "TF-IDF", "sbert": "SBERT", "gap": "Gap ($G$)"}


def r(path):
    return pd.read_csv(os.path.join(TAB, path))


def st(p):
    if pd.isna(p):
        return ""
    return "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else "^{\\dagger}" if p < .10 else ""


def num(x, m, p=None, dig=None):
    d = dig if dig is not None else (3 if m == "gap" else 4)
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return f"${s}{st(p) if p is not None else ''}$" if p is not None else f"${s}$"


def se(x, m, dig=None):
    d = dig if dig is not None else (3 if m == "gap" else 4)
    return f"({x:.{d}f})"


def pfmt(p):
    return "$<.001$" if p < .001 else f"${p:.3f}$".replace("0.", ".")


STARNOTE = ("$^{\\dagger}p<.10$. $^{*}p<.05$. $^{**}p<.01$. $^{***}p<.001$ "
            "(two-sided $t$ tests with 24 degrees of freedom).")


def table(label, caption, colspec, header, body, note, sideways=False, size="\\small", tight=False):
    env = "sidewaystable" if sideways else "table"
    out = [f"\\begin{{{env}}}[!htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", size]
    if tight:
        out.append("\\setlength{\\tabcolsep}{3pt}")
    out += [
           "\\begin{threeparttable}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out += header + ["\\midrule"] + body + ["\\bottomrule", "\\end{tabular}",
                                            "\\begin{tablenotes}[flushleft]", f"\\small\\item \\textit{{Note.}} {note}",
                                            "\\end{tablenotes}", "\\end{threeparttable}", f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def write(name, txt):
    open(os.path.join(OUTT, name), "w").write(txt)


# ---------------------------------------------------------------- Table 1 main
m = r("main_models.csv")
body = []
for i, d in enumerate(DOMAINS):
    if i:
        body.append("\\addlinespace")
    body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{DL[d]}}}}} \\\\")
    for k in MEASURES:
        x = m[(m.domain == d) & (m.measure == k)].iloc[0]
        body.append(f"\\quad {ML[k]} & {num(x.delta, k, x.p)} & {se(x.se, k)} & [{x.se_2way:.{3 if k=='gap' else 4}f}] "
                    f"& {num(x.beta, k, x.beta_p)} & {se(x.beta_se, k)} \\\\")
n = m.iloc[0]
write("tab1_main.tex", table(
    "tab:main", "Ideological-Distance Gradient Trend ($\\delta$) and Common Trend ($\\beta$) in Pairwise Similarity",
    "lccccc",
    ["& \\multicolumn{3}{c}{Model A: pair + month FE} & \\multicolumn{2}{c}{Model B: pair FE + trend} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
     "Domain and measure & $\\delta$ ($D\\times t$) & Dyadic \\textit{SE} & Two-way \\textit{SE} & $\\beta$ ($t$) & Dyadic \\textit{SE} \\\\"],
    body,
    "Estimates are from the primary sample (November 2015 to January 2023; " + f"{int(n.n):,}".replace(",", "{,}") + " pair-months per model, "
    f"{int(n.n_pairs)} outlet pairs, {int(n.n_months)} months). $D$ = absolute difference in audience slant, mean-centered; "
    "$t$ = years. $\\delta<0$ would indicate that ideologically distant pairs lose similarity faster than close pairs. "
    "$\\beta$ is the linear trend for a pair of average ideological distance. Dyadic cluster-robust standard errors "
    "\\parencite{aronow2015cluster} in parentheses; two-way (outlet $i$, outlet $j$) clustered standard errors in brackets. "
    "Model B estimates of $\\delta$ are within 0.0005 of Model A for every outcome (see supplementary code). "
    "Stars refer to dyadic standard errors. " + STARNOTE))

# ---------------------------------------------------------------- Table 2 H3
h = r("h3_pooled.csv")
body = []
for std in (False, True):
    body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{'Standardized within domain' if std else 'Raw similarity scores'}}}}} \\\\")
    for k in (["tfidf", "sbert", "gap"] if not std else ["tfidf", "sbert"]):
        x = h[(h.measure == k) & (h.standardised == std)].iloc[0]
        dg = 3 if (k == "gap" or std) else 4
        body.append(f"\\quad {ML[k]} & {num(x.delta_nonpol, k, None, dg)} {se(x.se_nonpol, k, dg)} & "
                    f"{num(x.delta_pol, k, None, dg)} {se(x.se_pol, k, dg)} & {num(x['diff'], k, x.diff_p, dg)} & "
                    f"{se(x.diff_se, k, dg)} & {pfmt(x.diff_p_2way)} \\\\")
    if not std:
        body.append("\\addlinespace")
write("tab2_h3.tex", table(
    "tab:h3", "Pooled Test of Domain Differences in the Distance-Gradient Trend (H3)",
    "lccccc",
    ["Measure & $\\delta_{\\text{non-pol}}$ (\\textit{SE}) & $\\delta_{\\text{pol}}$ (\\textit{SE}) & "
     "$D\\times t\\times$Political & Dyadic \\textit{SE} & $p$ (two-way) \\\\"],
    body,
    "Stacked political and non-political pair-months (50{,}050 observations, 300 pairs, 87 months) with pair $\\times$ domain "
    "and month $\\times$ domain fixed effects. The triple interaction is the difference between the political and "
    "non-political gradient trends; H3 predicts a negative value. The $G$ measure is already standardized. " + STARNOTE))

# ---------------------------------------------------------------- Table 3 asymmetry
a = r("asymmetry_midpoint.csv"); iso = r("isolation_models.csv"); b = r("bloc_segregation_trends.csv")
body = []
for i, d in enumerate(DOMAINS):
    if i:
        body.append("\\addlinespace")
    body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{DL[d]}}}}} \\\\")
    for k in MEASURES:
        x = a[(a.domain == d) & (a.measure == k)].iloc[0]
        y = iso[(iso.domain == d) & (iso.measure == k)].iloc[0]
        z = b[(b.domain == d) & (b.measure == k)].iloc[0]
        body.append(f"\\quad {ML[k]} & {num(x.mu, k, x.mu_p)} {se(x.mu_se, k)} & {num(y.lam, k, y.p)} {se(y.se, k)} & "
                    f"{num(z.LR_vs_LL, k, z.LR_p)} {se(z.LR_se, k)} & {num(z.RR_vs_LL, k, z.RR_p)} {se(z.RR_se, k)} \\\\")
write("tab3_asymmetry.tex", table(
    "tab:asym", "Asymmetry: Pair Midpoint, Outlet Isolation, and Bloc-Pair Trends",
    "lcccc",
    ["& (1) Pair & (2) Outlet & \\multicolumn{2}{c}{(3) Trend vs.\\ left--left pairs} \\\\",
     "\\cmidrule(lr){4-5}",
     "Domain and measure & $M\\times t$ & Slant $\\times\\, t$ & Left--right $\\times\\, t$ & Right--right $\\times\\, t$ \\\\"],
    body,
    "(1) Model A augmented with the mean-centered pair midpoint $M_{ij}=(s_i+s_j)/2$ interacted with $t$ (controls for "
    "$D\\times t$); positive values mean pairs with a more right-leaning midpoint gain similarity. Dyadic standard errors. "
    "(2) Outlet-month panel of each outlet's mean similarity to all other outlets present that month (2{,}130 outlet-months), "
    "with outlet and month fixed effects; negative values would mean right-leaning outlets become more isolated. Standard "
    "errors clustered by outlet (25 clusters). (3) Pair and month fixed effects with trends for left--right and right--right "
    "pairs (reference: left--left); right bloc = slant $>3$ (five outlets). Dyadic standard errors; for three cells the "
    "covariance matrix was made positive semi-definite by eigenvalue clipping \\parencite{cameron2011robust}. " + STARNOTE,
    size="\\footnotesize"))

# ---------------------------------------------------------------- Appendix tables
# A1 gaps
g = r("inspection_gaps.csv")
body = [f"@{o} & {s} & {e} & {n} \\\\" for o, s, e, n in g[["outlet", "start", "end", "months"]].values]
write("tabA1_gaps.tex", table(
    "tab:gaps", "Months in Which an Outlet Is Absent From the Similarity Data", "lccc",
    ["Outlet (account) & First month & Last month & Months \\\\"], body,
    "An outlet-month is absent when no pair involving that outlet is recorded; there are no partial (pair-level) gaps. "
    "The first Breitbart gap precedes the primary analysis window."))

# A2 descriptives (primary window vs pre-break)
d0, _ = load_raw()
cols = ["TF-IDF-Pol", "SBERT-Pol", "TF-IDF-NonPol", "SBERT-NonPol", "TF-IDF-All", "SBERT-All"]
body = []
for lab, x in [("Nov 2014--Oct 2015", d0[d0.Month < PRIMARY_START]), ("Nov 2015--Jan 2023 (primary)", d0[d0.Month >= PRIMARY_START])]:
    body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab}; $N={len(x):,}$ pair-months}}}} \\\\".replace(",", "{,}"))
    for c in cols:
        v = x[c]
        body.append(f"\\quad {c} & {v.mean():.3f} & {v.std():.3f} & {v.min():.3f} & {v.quantile(.25):.3f} & {v.median():.3f} & {v.max():.3f} \\\\")
    body.append("\\addlinespace")
body = body[:-1]
write("tabA2_descriptives.tex", table(
    "tab:desc", "Descriptive Statistics of the Six Pairwise Similarity Measures Before and After the October--November 2015 Break",
    "lcccccc", ["Measure & \\textit{M} & \\textit{SD} & Min & $Q_1$ & \\textit{Mdn} & Max \\\\"], body,
    "No values fall outside $[0,1]$ and none are missing. Before November 2015 TF-IDF similarities are compressed near 1 "
    "(cross-pair \\textit{SD} within month $\\approx .005$--.02), whereas SBERT shows no discontinuity (see Figure~\\ref{fig:raw})."))

# A3 robustness delta (sideways)
rb = r("robustness_delta.csv"); rh = r("robustness_h3.csv")
order = [("primary", "Primary (Nov 2015--Jan 2023)"), ("pre_break_incl", "Including pre-break months (from Nov 2014)"),
         ("to_2022_10", "Ending October 2022"), ("rank_D", "Rank distance instead of $D$"),
         ("interpolated", "Gaps linearly interpolated"), ("drop_gap", "Excluding Breitbart and Fox News"),
         ("pre_280", "Before November 2017 only"), ("post_280", "November 2017 onward only")]
cells = [(d, k) for d in DOMAINS for k in MEASURES]
body = []
for s, lab in order:
    x = rb[rb["sample"] == s]
    row1 = [num(x[(x.domain == d) & (x.measure == k)].iloc[0].delta, k, x[(x.domain == d) & (x.measure == k)].iloc[0].p) for d, k in cells]
    row2 = [se(x[(x.domain == d) & (x.measure == k)].iloc[0].se, k) for d, k in cells]
    body.append(f"{lab} & " + " & ".join(row1) + f" & {int(x.iloc[0].n_pairs)} & {int(x.iloc[0].n_months)} \\\\")
    body.append(" & " + " & ".join(row2) + " & & \\\\")
write("tabA3_robust_delta.tex", table(
    "tab:robdelta", "Robustness of the Distance-Gradient Trend $\\delta$ (Model A)",
    "l" + "c" * 11,
    ["& \\multicolumn{3}{c}{Political} & \\multicolumn{3}{c}{Non-political} & \\multicolumn{3}{c}{All posts} & & \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Specification & TF-IDF & SBERT & $G$ & TF-IDF & SBERT & $G$ & TF-IDF & SBERT & $G$ & Pairs & Months \\\\"],
    body,
    "Each cell is $\\delta$ from a separate Model A regression (pair and month fixed effects) with its dyadic standard error "
    "in parentheses. Rank-distance estimates are per slant rank (1--24) and are not on the same scale as the other rows. "
    "In the sub-period and trimmed samples $G$ is re-standardized within the sample. " + STARNOTE,
    sideways=True, size="\\footnotesize", tight=True))

# A4 robustness beta
body = []
for s, lab in order:
    if s == "rank_D":
        continue
    x = rb[rb["sample"] == s]
    r1, r2 = [], []
    for d in DOMAINS:
        for k in ["tfidf", "sbert"]:
            y = x[(x.domain == d) & (x.measure == k)].iloc[0]
            r1.append(num(y.beta, k, y.beta_p)); r2.append(se(y.beta_se, k))
    body.append(f"{lab} & " + " & ".join(r1) + " \\\\")
    body.append(" & " + " & ".join(r2) + " \\\\")
write("tabA4_robust_beta.tex", table(
    "tab:robbeta", "Robustness of the Common Linear Trend $\\beta$ (Model B)",
    "lcccccc",
    ["& \\multicolumn{2}{c}{Political} & \\multicolumn{2}{c}{Non-political} & \\multicolumn{2}{c}{All posts} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
     "Specification & TF-IDF & SBERT & TF-IDF & SBERT & TF-IDF & SBERT \\\\"],
    body, "$\\beta$ is the linear time trend (per year) for a pair of average ideological distance, from models with pair "
    "fixed effects and $D\\times t$. Dyadic standard errors in parentheses. " + STARNOTE, sideways=True, size="\\footnotesize"))

# A5 H3 robustness
body = []
for s, lab in order:
    x = rh[(rh["sample"] == s) & (~rh.standardised)]
    r1 = []; r2 = []
    for k in MEASURES:
        y = x[x.measure == k].iloc[0]
        r1.append(num(y["diff"], k, y.diff_p)); r2.append(se(y.diff_se, k))
    body.append(f"{lab} & " + " & ".join(r1) + " \\\\")
    body.append(" & " + " & ".join(r2) + " \\\\")
write("tabA5_robust_h3.tex", table(
    "tab:robh3", "Robustness of the Pooled Domain-Difference Test (H3): $D\\times t\\times$Political",
    "lccc", ["Specification & TF-IDF & SBERT & $G$ \\\\"], body,
    "Coefficient on the triple interaction in the stacked political and non-political model with pair $\\times$ domain and "
    "month $\\times$ domain fixed effects; dyadic standard errors in parentheses. Negative values would support H3. " + STARNOTE,
    size="\\footnotesize"))

# A6 280-character interaction
x280 = r("robust_280_interaction.csv")
body = []
for i, d in enumerate(DOMAINS):
    if i:
        body.append("\\addlinespace")
    body.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{DL[d]}}}}} \\\\")
    for k in MEASURES:
        y = x280[(x280.domain == d) & (x280.measure == k)].iloc[0]
        body.append(f"\\quad {ML[k]} & {num(y.delta, k, y.delta_p)} {se(y.delta_se, k)} & {num(y.D_post, k, y.D_post_p)} {se(y.D_post_se, k)} \\\\")
write("tabA6_280.tex", table(
    "tab:280", "Distance Gradient and the November 2017 Character-Limit Change", "lcc",
    ["Domain and measure & $D\\times t$ & $D\\times$Post-Nov 2017 \\\\"], body,
    "Model A augmented with $D$ interacted with an indicator for months from November 2017 onward (the 280-character limit). "
    "Dyadic standard errors in parentheses. " + STARNOTE))

# A7 bloc segregation
bm = r("bloc_segregation_trends_median.csv")
body = []
for i, d in enumerate(DOMAINS):
    if i:
        body.append("\\addlinespace")
    body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{DL[d]}}}}} \\\\")
    for k in MEASURES:
        y = b[(b.domain == d) & (b.measure == k)].iloc[0]; z = bm[(bm.domain == d) & (bm.measure == k)].iloc[0]
        body.append(f"\\quad {ML[k]} & {num(y.series_slope, k, y.series_p)} {se(y.series_se_nw, k)} & "
                    f"{num(y.phi_between_t, k, y.phi_p)} {se(y.phi_se, k)} & "
                    f"{num(z.series_slope, k, z.series_p)} {se(z.series_se_nw, k)} & "
                    f"{num(z.phi_between_t, k, z.phi_p)} {se(z.phi_se, k)} \\\\")
write("tabA7_bloc.tex", table(
    "tab:bloc", "Trends in Bloc Segregation Under Two Bloc Definitions", "lcccc",
    ["& \\multicolumn{2}{c}{Midpoint rule (slant $>3$)} & \\multicolumn{2}{c}{Median split (slant $>2.62$)} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     "Domain and measure & Series trend & Between $\\times\\, t$ & Series trend & Between $\\times\\, t$ \\\\"], body,
    "Series trend: OLS slope (per year) of the monthly difference between mean within-bloc and mean between-bloc similarity, "
    "Newey--West standard errors (6 lags); positive values indicate rising segregation. Between $\\times\\, t$: pair-month "
    "model with pair and month fixed effects, which is robust to changes in bloc composition when outlets are absent; "
    "negative values indicate rising segregation; dyadic standard errors. Under the median split the dyadic covariance "
    "was not positive semi-definite for eight of the nine cells, so the conservative estimator was used (see Section 2). " + STARNOTE,
    sideways=True, size="\\footnotesize"))

# A8 year gradient coefficients (sideways)
yg = r("year_gradient.csv")
years = sorted(yg.year.unique())
body = []
for y in years:
    if y == 2016:
        body.append("2016 (reference) & " + " & ".join(["---"] * 9) + " \\\\")
        continue
    r1, r2 = [], []
    for d, k in cells:
        z = yg[(yg.domain == d) & (yg.measure == k) & (yg.year == y)].iloc[0]
        r1.append(num(z.coef, k, z.p, 3)); r2.append(se(z.se, k, 3))
    lab = str(y) + (" (incl. Jan 2023)" if y == 2022 else "")
    body.append(f"{lab} & " + " & ".join(r1) + " \\\\")
    body.append(" & " + " & ".join(r2) + " \\\\")
write("tabA8_year_gradient.tex", table(
    "tab:yeargrad", "Year-Specific Distance Gradients Relative to 2016 ($\\theta_y$), Estimates Underlying Figure 2 of the Main Text",
    "l" + "c" * 9,
    ["& \\multicolumn{3}{c}{Political} & \\multicolumn{3}{c}{Non-political} & \\multicolumn{3}{c}{All posts} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Year & TF-IDF & SBERT & $G$ & TF-IDF & SBERT & $G$ & TF-IDF & SBERT & $G$ \\\\"], body,
    "Coefficients on $D\\times\\mathbb{1}[\\text{year}=y]$ from models with pair and month fixed effects; the 2016 reference "
    "bin also contains November and December 2015. Dyadic standard errors in parentheses (eigenvalue-clipped where "
    "necessary). " + STARNOTE, sideways=True, size="\\footnotesize", tight=True))

# A9 leave-one-out summary
loo = r("loo_political.csv")
body = []
for k in MEASURES:
    z = loo[loo.measure == k]; full = z[z.dropped == "(none)"].iloc[0]; o = z[z.dropped != "(none)"]
    mn, mx = o.loc[o.delta.idxmin()], o.loc[o.delta.idxmax()]
    body.append(f"{ML[k]} & {num(full.delta, k)} & {num(mn.delta, k)} (@{mn.dropped}) & {num(mx.delta, k)} (@{mx.dropped}) & "
                f"{(o.delta > 0).sum()}/25 & {(o.p < .05).sum()}/25 \\\\")
write("tabA9_loo.tex", table(
    "tab:loo", "Leave-One-Outlet-Out Estimates of the Political Distance-Gradient Trend", "lccccc",
    ["Measure & Full sample $\\delta$ & Minimum (outlet dropped) & Maximum (outlet dropped) & $\\delta>0$ & $p<.05$ \\\\"],
    body, "Model A for political content re-estimated 25 times, each time excluding every pair that involves one outlet. "
    "Full distributions are shown in Figure~\\ref{fig:loo}.", size="\\footnotesize"))

print("written:", sorted(os.listdir(OUTT)))
