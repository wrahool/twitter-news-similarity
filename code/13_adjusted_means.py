"""
13_adjusted_means.py -- cross-sectional effect of ideological distance using
month-adjusted pair means instead of raw pair means.

Raw pair means average each pair over the months it is observed, so pairs with
gaps (Fox, Breitbart, Politico, VICE) are averaged over a different set of
months while similarity is trending down. Here each pair's level is instead the
pair fixed effect from  Sim_ijt = a_ij + g_t + e  (two-way FE, no distance
term), i.e. its average similarity net of month-specific shocks, on a common
baseline (re-centred to the grand mean so the scale matches the raw means).

Then: a_ij ~ D (no controls) and a_ij ~ D + outlet-type counts (effect-coded,
as in 10_type_regression.py). Dyadic SEs. Slant: Ad Fontes (default here) and
Pew for comparison. Scores: average of the five samples; window Nov 2015 - Jan 2023.
Note: SEs treat the estimated a_ij as data (each is based on up to 87 months,
so estimation error is small relative to between-pair variation).

Outputs: tables/adjusted_means_effects.csv, tables/adjusted_pair_means.csv
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import *

fl = importlib.import_module("07_first_last")
tr = importlib.import_module("10_type_regression")
AVG = os.path.join(OUT, "multisample", "sample_avg_monthly_pairwise_similarity.csv")


def pair_effects(x, y):
    """Pair fixed effects from y = a_pair + g_month (alternating projections)."""
    pc, pairs = pd.factorize(x.pair)
    mc, _ = pd.factorize(x.month)
    v = x[y].values
    a = np.zeros(pc.max() + 1); g = np.zeros(mc.max() + 1)
    for _ in range(10000):
        a_old = a.copy()
        g = np.bincount(mc, weights=v - a[pc]) / np.bincount(mc)
        g -= g.mean()                                   # normalise month effects to mean zero
        a = np.bincount(pc, weights=v - g[mc]) / np.bincount(pc)
        if np.max(np.abs(a - a_old)) < 1e-13:
            break
    return pd.Series(a, index=pairs)


if __name__ == "__main__":
    common.DATA_SIM = AVG
    rows, means = [], []
    for src in ["adfontes", "pew"]:
        common.SLANT_SOURCE = src
        p, _ = build_panel()
        keys = p.groupby("pair")[["o1", "o2", "D"]].first()
        for d in DOMAINS:
            x = p[p.domain == d]
            for m in ["tfidf", "sbert"]:
                adj = pair_effects(x, m)
                raw = x.groupby("pair")[m].mean()
                df = keys.copy()
                df["adj"] = adj.reindex(df.index)
                df["adj"] += raw.mean() - df.adj.mean()          # same overall level as raw means
                df["raw"] = raw.reindex(df.index)
                df = df.reset_index()
                if src == "adfontes":
                    t = df.assign(domain=d, measure=m, n_months=x.groupby("pair").month.nunique().reindex(df.pair).values)
                    means.append(t)
                sdD, sdY = df.D.std(), df.adj.std()
                for outcome in ["raw", "adj"]:
                    b, se, pv, r = fl.ols_dyadic(df, outcome)
                    rows.append(dict(source=src, domain=d, measure=m, outcome=outcome, controls="none",
                                     b=b[1], se=se[1], p=pv[1], r2=r ** 2,
                                     per_sd_D=b[1] * sdD, std_beta=b[1] * sdD / df[outcome].std(),
                                     full_range=b[1] * df.D.max()))
                    f = tr.fit(df.rename(columns={outcome: "y"}), "y")
                    fr = f[f.term.str.startswith("Ideological")].iloc[0]
                    rows.append(dict(source=src, domain=d, measure=m, outcome=outcome, controls="format + UK",
                                     b=fr.est, se=fr.se, p=fr.p, r2=fr.r2,
                                     per_sd_D=fr.est * sdD, std_beta=fr.est * sdD / df[outcome].std(),
                                     full_range=fr.est * df.D.max(), vcov_fix=fr.vcov_fix))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TAB, "adjusted_means_effects.csv"), index=False)
    mm = pd.concat(means)
    mm.to_csv(os.path.join(TAB, "adjusted_pair_means.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.round(4).to_string(index=False))
    # which pairs moved most between raw and adjusted means (political TF-IDF)
    q = mm[(mm.domain == "pol") & (mm.measure == "tfidf")].assign(diff=lambda z: z.adj - z.raw)
    print(q.sort_values("diff")[["pair", "n_months", "raw", "adj", "diff"]].head(6).round(4).to_string(index=False))
    print(q.sort_values("diff")[["pair", "n_months", "raw", "adj", "diff"]].tail(6).round(4).to_string(index=False))
    print("corr raw/adj by domain-measure:", mm.groupby(["domain", "measure"]).apply(lambda z: z.raw.corr(z.adj)).round(4).to_dict())
