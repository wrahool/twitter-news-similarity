"""
02_main_models.py -- Step 4: main models, linear-trend models, H3 pooled test.

Model A (month FE):   Sim_ijt = a_ij + g_t + delta * (Dc_ij x t) + e
Model B (linear):     Sim_ijt = a_ij + beta * t + delta * (Dc_ij x t) + e
    Dc = D - mean(D), so beta is the common trend of an average-distance
    pair; delta is identical to the uncentred specification.
H3 pooled: stack political + non-political; FE are pair x domain and
    month x domain (full interaction of lower-order terms), plus
    Dc x t and Dc x t x Pol. The triple interaction tests whether the
    gradient steepens more for political content; stacking lets the dyadic
    SEs account for the correlation across domains within a dyad.
    Estimated on raw scores and on scores standardised within domain
    (SBERT-Pol has a much smaller spread than the other series).

Outputs: tables/main_models.csv, tables/table1_main.md/.tex,
         tables/h3_pooled.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def run_main(p, label="primary", dvar="D"):
    """Return long table of delta / beta for 9 domain x measure cells."""
    p = p.copy()
    pair_mean = p.groupby("pair")[dvar].first().mean()
    p["Dt"] = (p[dvar] - pair_mean) * p.t
    rows = []
    for dom in DOMAINS:
        x = p[p.domain == dom]
        for m in MEASURES:
            a = fe_ols(x, m, ["Dt"], ["pair", "month"])
            b = fe_ols(x, m, ["t", "Dt"], ["pair"])
            ta, tb = a["tab"], b["tab"]
            rows.append(dict(sample=label, domain=dom, measure=m,
                             delta=ta.loc["Dt", "coef"], se=ta.loc["Dt", "se_dyadic"],
                             se_2way=ta.loc["Dt", "se_twoway"], p=ta.loc["Dt", "p_dyadic"],
                             p_2way=ta.loc["Dt", "p_twoway"], lo=ta.loc["Dt", "lo"], hi=ta.loc["Dt", "hi"],
                             r2w=a["r2_within"],
                             beta=tb.loc["t", "coef"], beta_se=tb.loc["t", "se_dyadic"],
                             beta_se_2way=tb.loc["t", "se_twoway"], beta_p=tb.loc["t", "p_dyadic"],
                             delta_B=tb.loc["Dt", "coef"], delta_B_se=tb.loc["Dt", "se_dyadic"],
                             delta_B_p=tb.loc["Dt", "p_dyadic"],
                             n=a["n"], n_pairs=a["n_pairs"], n_months=a["n_months"],
                             sd_y=x[m].std()))
    return pd.DataFrame(rows)


def run_h3(p, standardise=False):
    p = p.copy()
    pair_mean = p.groupby("pair").D.first().mean()
    p["Dt"] = (p.D - pair_mean) * p.t
    s = p[p.domain.isin(["pol", "nonpol"])].copy()
    s["pol"] = (s.domain == "pol").astype(int)
    s["Dt_pol"] = s.Dt * s.pol
    s["pair_dom"] = s.pair + "#" + s.domain
    s["month_dom"] = s.month + "#" + s.domain
    rows = []
    for m in MEASURES:
        if standardise and m != "gap":
            s[m + "_z"] = s.groupby("domain")[m].transform(lambda v: (v - v.mean()) / v.std())
            y = m + "_z"
        else:
            y = m
        r = fe_ols(s, y, ["Dt", "Dt_pol"], ["pair_dom", "month_dom"])
        t = r["tab"]
        dp, dpse, dpp = lincom(r, {"Dt": 1, "Dt_pol": 1})
        rows.append(dict(measure=m, standardised=standardise,
                         delta_nonpol=t.loc["Dt", "coef"], se_nonpol=t.loc["Dt", "se_dyadic"],
                         delta_pol=dp, se_pol=dpse,
                         diff=t.loc["Dt_pol", "coef"], diff_se=t.loc["Dt_pol", "se_dyadic"],
                         diff_se_2way=t.loc["Dt_pol", "se_twoway"],
                         diff_p=t.loc["Dt_pol", "p_dyadic"], diff_p_2way=t.loc["Dt_pol", "p_twoway"],
                         n=r["n"], n_pairs=r["n_pairs"], n_months=r["n_months"]))
    return pd.DataFrame(rows)


def journal_table(res, path_md, path_tex, title):
    """Columns = domain x measure; rows = delta (month FE), beta and delta (linear trend)."""
    cols = [(d, m) for d in DOMAINS for m in MEASURES]
    head = [f"{DOMAIN_LABEL[d]}<br>{MEASURE_LABEL[m].split(' (')[0]}" for d, m in cols]
    lines = []
    def cell(d, m, k, sek, pk):
        r = res[(res.domain == d) & (res.measure == m)].iloc[0]
        return fmt(r[k], r[sek], r[pk])
    body = {
        "Model A (pair + month FE)": None,
        "D x t (delta)": ("delta", "se", "p"),
        "   two-way clustered SE": ("delta", "se_2way", "p_2way"),
        "Model B (pair FE + linear trend)": None,
        "t (beta; common trend at mean D)": ("beta", "beta_se", "beta_p"),
        "D x t (delta)  ": ("delta_B", "delta_B_se", "delta_B_p"),
    }
    md = [f"**{title}**", "", "| | " + " | ".join(head) + " |", "|---" * (len(cols) + 1) + "|"]
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DOMAIN_LABEL[d]}}}" for d in DOMAINS) + " \\\\",
           " & " + " & ".join(MEASURE_LABEL[m].split(" (")[0] for d, m in cols) + " \\\\", "\\midrule"]
    for name, spec in body.items():
        if spec is None:
            md.append(f"| *{name}* |" + " |" * len(cols))
            tex.append(f"\\multicolumn{{{len(cols)+1}}}{{l}}{{\\textit{{{name}}}}} \\\\")
            continue
        k, sek, pk = spec
        cs = [cell(d, m, k, sek, pk) for d, m in cols]
        if "two-way" in name:
            md.append(f"| {name} | " + " | ".join(f"[{c[1][1:-1]}]{stars(res[(res.domain==d)&(res.measure==m)].iloc[0]['p_2way'])}" for c, (d, m) in zip(cs, cols)) + " |")
            tex.append(name.strip() + " & " + " & ".join(f"[{c[1][1:-1]}]" for c in cs) + " \\\\")
        else:
            md.append(f"| {name.strip()} | " + " | ".join(c[0] for c in cs) + " |")
            md.append("| | " + " | ".join(c[1] for c in cs) + " |")
            tex.append(name.strip().replace("x", "$\\times$") + " & " + " & ".join(c[0].replace("*", "$^{*}$") for c in cs) + " \\\\")
            tex.append(" & " + " & ".join(c[1] for c in cs) + " \\\\")
    for lab, k in [("N pair-months", "n"), ("N pairs", "n_pairs"), ("N months", "n_months")]:
        vals = [str(int(res[(res.domain == d) & (res.measure == m)].iloc[0][k])) for d, m in cols]
        md.append(f"| {lab} | " + " | ".join(vals) + " |")
        tex.append(lab + " & " + " & ".join(vals) + " \\\\")
    md += ["", "Notes: Cells are OLS coefficients; dyadic cluster-robust standard errors "
           "(Aronow, Samii & Assenova 2015; 25 outlets) in parentheses; two-way (outlet-1, outlet-2) "
           "clustered SEs in brackets. D = |slant_i - slant_j| (Pew 1-5 audience scale), mean-centred; "
           "t in years. delta < 0 = the similarity gap between distant and close pairs widens over time. "
           "+ p<.10, * p<.05, ** p<.01, *** p<.001 (t with 24 df)."]
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(path_md, "w").write("\n".join(md))
    open(path_tex, "w").write("\n".join(tex))


if __name__ == "__main__":
    p, _ = build_panel()
    res = run_main(p)
    res.to_csv(os.path.join(TAB, "main_models.csv"), index=False)
    journal_table(res, os.path.join(TAB, "table1_main.md"), os.path.join(TAB, "table1_main.tex"),
                  "Table 1. Ideological-distance gradient and common trend in pairwise similarity, 2015-11 to 2023-01")
    h3 = pd.concat([run_h3(p, False), run_h3(p, True)])
    h3.to_csv(os.path.join(TAB, "h3_pooled.csv"), index=False)
    save_fallback_log("main")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res[["domain", "measure", "delta", "se", "se_2way", "p", "beta", "beta_se", "beta_p", "delta_B", "n", "sd_y"]].round(5))
    print(h3.round(5))
