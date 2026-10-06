"""
run_pipeline.py -- run every analysis, regressions and network, in order, headless.

Stages (run in this order; pick a subset with --stages):
  trend      01-04, 06-14 on the UNCENTRED-SBERT samples (samples/samples data uncentred)
             -> results/uncentred/. Source of every common-trend / H1 result and of
             the averaged uncentred file that 16 and 18 read.
  main       01-04, 06-14 on the centred-SBERT samples (samples/samples data centred) -> results/.
             Source of the level, distance-gradient, H2, H3, asymmetry and bloc results.
  postcount  15 (post-count checks), 16 (trend models with post-count controls)
  network    17 (communities, ideology alignment, yearly bloc modularity),
             17b (yearly modularity with Fox / Breitbart / Politico+VICE / all four dropped)
  robust     18 (permutation / bootstrap / jackknife / Driscoll-Kraay inference),
             19 (start vs end network comparison) for Pew and Ad Fontes slant

Usage:
  python run_pipeline.py                         # everything, all cores
  python run_pipeline.py --quick                 # smoke test of everything (~30-60 min)
  python run_pipeline.py --stages main,postcount # a subset
  python run_pipeline.py --jobs 16 --keep-going  # limit workers; continue past a failed step
Log: results/logs/pipeline_<timestamp>.log (plus each step's own console output).
"""
import os, sys, time, argparse, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
CORE = ["01_inspect.py", "02_main_models.py", "03_extensions.py", "04_robustness.py", "06_multisample.py",
        "07_first_last.py", "08_outlet_type.py", "09_outlet_type_plot.py", "10_type_regression.py",
        "11_heatmap.py", "12_adfontes.py", "13_adjusted_means.py", "14_sbert_distribution.py"]
STAGES = ["trend", "main", "postcount", "network", "robust"]


def steps(stage, a):
    jobs = ["--jobs", str(a.jobs)]
    q = ["--quick"] if a.quick else []
    if stage == "trend":
        env = {"SAMPLES_DIR": a.uncentred_dir, "OUT_DIR": os.path.join(RESULTS, "uncentred")}
        return [([s], env) for s in CORE]
    if stage == "main":
        return [([s], {}) for s in CORE]
    if stage == "postcount":
        return [(["15_postcount_check.py"], {}), (["16_postcount_trends.py"], {})]
    if stage == "network":
        drops = ["FoxNews", "BreitbartNews", "politico,VICE", "FoxNews,BreitbartNews,politico,VICE"]
        return [(["17_network.py"], {})] + [(["17b_network_yearly_drop.py", d], {}) for d in drops]
    if stage == "robust":
        return [(["18_robust_inference.py"] + jobs + q, {}),
                (["19_network_compare.py", "--source", "pew"] + jobs + q, {}),
                (["19_network_compare.py", "--source", "adfontes"] + jobs + q, {})]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--jobs", type=int, default=-1, help="workers for 18/19 (-1 = all cores)")
    ap.add_argument("--quick", action="store_true", help="few permutations / bootstraps everywhere")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--uncentred-dir", default=os.path.join("samples", "samples data uncentred"),
                    help="folder (relative to the project) with the uncentred-SBERT sample files")
    a = ap.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s) {bad}; choose from {STAGES}")

    # input checks
    need = [os.path.join(ROOT, "auxiliary", f) for f in ["media_slant.csv", "media_slant2.csv", "sampling_audit.csv"]]
    need += [os.path.join(ROOT, "samples", "samples data centred", f"sample{k}_monthly_pairwise_similarity.csv")
             for k in range(1, 6)]
    if "trend" in stages:
        need += [os.path.join(ROOT, a.uncentred_dir, f"sample{k}_monthly_pairwise_similarity.csv") for k in range(1, 6)]
    missing = [n for n in need if not os.path.exists(n)]
    if missing:
        sys.exit("missing input files:\n  " + "\n  ".join(missing))
    if "trend" not in stages and any(s in stages for s in ["postcount", "robust"]):
        unc = os.path.join(RESULTS, "uncentred", "multisample", "sample_avg_monthly_pairwise_similarity.csv")
        if not os.path.exists(unc):
            sys.exit("results/uncentred not found: run the trend stage first (16 and 18 need it)")

    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(RESULTS, "logs", f"pipeline_{stamp}.log")
    base_env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if a.quick:
        base_env["QUICK"] = "1"
    summary, t_all = [], time.time()
    with open(logpath, "w") as log:
        for st in stages:
            for args, extra in steps(st, a):
                env = dict(base_env, **extra)
                head = f"[{datetime.datetime.now():%H:%M:%S}] stage {st}: {' '.join(args)}" + \
                       (f"  ({', '.join(f'{k}={v}' for k, v in extra.items())})" if extra else "")
                print(head, flush=True); log.write(head + "\n"); log.flush()
                t0 = time.time()
                proc = subprocess.Popen([sys.executable] + args, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    log.write(line)
                log.flush()
                rc = proc.wait()
                dt = time.time() - t0
                status = "ok" if rc == 0 else f"FAILED (exit {rc})"
                msg = f"    -> {status} in {dt / 60:.1f} min"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                summary.append((st, " ".join(args), status, dt))
                if rc != 0 and not a.keep_going:
                    print(f"stopped; see {logpath}", flush=True)
                    sys.exit(rc)
        tail = f"\nfinished in {(time.time() - t_all) / 60:.1f} min; " \
               f"{sum(s[2] != 'ok' for s in summary)} failed step(s). Log: {logpath}"
        print(tail, flush=True); log.write(tail + "\n")
        for st, cmd, status, dt in summary:
            log.write(f"{st:10s} {status:18s} {dt / 60:6.1f} min  {cmd}\n")


if __name__ == "__main__":
    main()
