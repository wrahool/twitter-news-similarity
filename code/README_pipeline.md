# News similarity project: full analysis pipeline

One command runs every analysis in this project, regressions and network work alike, on a headless Linux server:

```bash
python run_pipeline.py
```

## 1. What the server needs

Copy these three folders and keep the layout. Paths are resolved relative to `code/`.

```
news similarity project/                 (any name)
├── auxiliary/
│   ├── media_slant.csv                  Pew audience slant (1–5)
│   ├── media_slant2.csv                 Ad Fontes content score
│   └── sampling_audit.csv               post counts per outlet × month × condition
├── samples/
│   ├── samples data centred/            sample1..5_monthly_pairwise_similarity.csv  (centred SBERT)
│   └── samples data uncentred/          sample1..5_monthly_pairwise_similarity.csv  (uncentred SBERT)
└── code/                                this folder
```

The pipeline creates `results/` next to them and writes everything there. The trend stage writes to `results/uncentred/`.

TF-IDF is identical in the two sample folders; only the SBERT columns differ. You need both sets:

- **Centred SBERT** removes each month's shared direction, which is what makes topic differences between outlets visible.
- **Uncentred SBERT** is needed for every common-trend estimate (H1), because centring removes the common trend by construction.

## 2. Setup (Python ≥ 3.10)

```bash
cd "news similarity project/code"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If the scripts were copied from Windows and `bash` complains about `\r`, run `dos2unix *.sh *.py`. If `dos2unix` isn't installed, use `sed -i 's/\r$//' *.sh *.py`.

## 3. Run

```bash
python run_pipeline.py --quick                  # smoke test of everything first (~30–60 min on 2 cores)
python run_pipeline.py                          # full run, all cores
python run_pipeline.py --jobs 16                # limit workers (used by 18 and 19)
python run_pipeline.py --stages robust          # only some stages
python run_pipeline.py --keep-going             # don't stop at a failed step
nohup python run_pipeline.py > pipeline.out 2>&1 &   # detach from the terminal
```

Progress goes to the console. Full output goes to `results/logs/pipeline_<timestamp>.log`, which ends with a per-step summary.

| Stage | Scripts | Data | Writes to |
|---|---|---|---|
| `trend` | 01–04, 06–14 | uncentred SBERT samples | `results/uncentred/` |
| `main` | 01–04, 06–14 | centred SBERT samples | `results/` |
| `postcount` | 15, 16 | both (averaged files) | `results/` |
| `network` | 17, 17b (×4 drop sets) | centred | `results/` |
| `robust` | 18, then 19 (Pew), then 19 (Ad Fontes) | both | `results/` |

Stages depend on earlier ones:
- `postcount`, `network` and `robust` read the averaged files written by `06_multisample.py`.
- 16 and 18 read the uncentred averaged file from the `trend` stage (`results/uncentred/multisample/`). Run `trend` at least once before them.

**Runtime.** These figures are rough estimates. 18 is by far the most expensive script, and its runtime scales with `--n-perm` and `--n-boot` (defaults 1999 and 999):
- 18: about 75 estimates × ~10 CPU-minutes each, i.e. 1–2 h on 8 cores, or about 30 min on 32.
- Everything else together: under an hour.

To change the defaults, run the scripts directly. Each has `--help`, for example:

```bash
python 18_robust_inference.py --jobs 32 --n-perm 4999 --n-boot 1999
python 19_network_compare.py --period-a 2014-11:2017-10 --period-b 2021-01:2023-01 --source pew
```

## 4. Which output answers which question

Analytic choices are documented at the top of `common.py` and of each script.

| Question | Main output (in `results/tables/` and `results/figures/`) |
|---|---|
| H1: did non-political coverage converge? | `results/uncentred/tables/multisample_summary.csv` (family `main_beta`, domain `nonpol`), `postcount_trends.csv`, and `robust_inference.csv` (rows "H1 beta"; use the `p_dyadic_time` column) |
| Is ideology related to similarity (levels)? | `adjusted_means_effects.csv`, `type_regression.csv`, `postcount_robustness.csv` ("level gradient"), `robust_inference.csv` ("Level gradient"), heatmaps |
| H2: did distant pairs diverge? | `multisample_summary.csv` (`main_delta`), `year_gradient.csv`, `postcount_robustness.csv`, `robust_inference.csv` ("H2 delta") |
| H3: more divergence in political content? | `multisample_summary.csv` (`h3_diff`), `robust_inference.csv` ("H3") |
| RQ: did right-leaning outlets drive divergence? | `asymmetry_midpoint.csv`, `bloc_segregation_trends.csv`, `isolation_models.csv`, `robust_inference.csv` ("RR x t", "M x t") |
| Slant measure robustness (Ad Fontes) | `adfontes_*.csv`, `figS10_adfontes_vs_pew` |
| Sample-size artefacts | `postcount_mechanical.csv`, `postcount_robustness.csv`, `postcount_trends.csv` |
| Network communities and ideology | `network_tests.csv`, `network_communities.csv`, `fig_network` |
| Did the left/right split sharpen? Who drove it? | `network_yearly.csv`, `network_yearly_drop_*.csv`, `figS_network_yearly` |
| Start vs end: did the structure change, who moved? | `netcmp_qap.csv`, `netcmp_mrqap.csv`, `netcmp_outlets.csv`, `netcmp_global.csv`, `netcmp_multilayer*.csv`, `fig_network_change`, `figS_network_alluvial` (files with `_adfontes` use Ad Fontes slant) |
| Is inference robust to network dependence? | `robust_inference.csv`, `figS_robust_inference` |

### Reading `robust_inference.csv`

| Column | What it is | Use it for |
|---|---|---|
| `p_dyadic` | Analytic dyadic cluster-robust SE (Aronow, Samii & Assenova 2015), as used throughout | Reference |
| `p_dyadic_time` | Dyadic plus Driscoll–Kraay, trend rows only. Also allows for shocks shared by all pairs in a month. | The p-value to report for H1 |
| `p_perm_t` | Outlet-permutation test of the studentised statistic | The most assumption-free test for slant terms |
| `p_boot` | Node (pigeonhole) bootstrap; conservative | A result significant here is solid |
| `p_jackknife` | Delete-one-outlet jackknife | Further check |

### Reading `netcmp_outlets.csv`

- `drift_right` is how far an outlet moved towards the right bloc between the two periods.
- `drift_minus_median` and `drift_q` test whether it moved more than the median outlet. `drift_q` is BH-corrected across outlets.

## 5. Notes

- **Seeds.** Every script fixes its seeds, so reruns with the same settings reproduce the results.
- **`05_latex_tables.py` is not run.** Its table notes still describe the v1 analysis. Regenerate the manuscript tables once the results are final.
- **Bayesian layered SBM not included.** graph-tool installs only through conda (`conda install -c conda-forge graph-tool`) and could not be tested, so this model was left out.
