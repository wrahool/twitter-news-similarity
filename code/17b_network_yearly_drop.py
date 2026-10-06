# 17b: yearly bloc modularity with given outlets dropped. Usage: python 17b_network_yearly_drop.py FoxNews  (comma-separated list)
import sys, os, importlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import common
from common import *
net = importlib.import_module("17_network"); pc = net.pc
common.DATA_SIM = net.AVG
rows = []
DROP = set(sys.argv[1].split(","))
for src in ["pew", "adfontes"]:
    common.SLANT_SOURCE = src
    p, slant = build_panel(); p = pc.add_counts(p); cut = BLOC_CUT[src]
    p = p[~p.o1.isin(DROP) & ~p.o2.isin(DROP)]
    for dom in ["pol", "nonpol", "all"]:
        x = p[p.domain == dom]; ctrl = [] if dom == "all" else pc.CTRL
        for m in ["tfidf", "sbert"]:
            for yr in range(2015, 2023):
                xy = x[x.year.clip(2015, 2022) == yr]
                G = net.graph(net.pair_levels(xy, m, ctrl), m)
                nodes = sorted(G.nodes); s = np.array([slant[n] for n in nodes]); bloc = (s > cut).astype(int)
                qb = net.q_of(G, bloc, nodes)
                qn = np.array([net.q_of(G, bloc[net.RNG.permutation(len(nodes))], nodes) for _ in range(net.NYPERM)])
                rows.append(dict(source=src, domain=dom, measure=m, year=yr, n=len(nodes), Q_bloc=qb, p=(1+(qn>=qb).sum())/(1+len(qn))))
r = pd.DataFrame(rows); r.to_csv(os.path.join(TAB, 'network_yearly_drop_' + sys.argv[1].replace(',', '_') + '.csv'), index=False)
pd.set_option("display.width", 250)
print(r.pivot_table(index=["source","domain","measure"], columns="year", values="Q_bloc").round(3).to_string())
print(r.pivot_table(index=["source","domain","measure"], columns="year", values="p").round(3).to_string())
print(r.groupby("year").n.agg(["min","max"]).T)
