"""Compare DR methods: standalone vs AFE-enhanced, across multiple backends.

Composable multi-backend benchmark runner with JSONL resumability,
optimal arrow-count selection, and comprehensive significance testing.

Usage:
    python benchmarks/compare_methods.py --category all --n-seeds 10
    python benchmarks/compare_methods.py --datasets mnist --backends pacmap,umap --n-seeds 3
    python benchmarks/compare_methods.py --category scrna --n-arrows 47
    python benchmarks/compare_methods.py --n-arrows sweep  # exploratory multi-count
    python benchmarks/compare_methods.py --list-datasets
"""

# CRITICAL: Prevent OpenBLAS threading deadlock on Windows.
# Must be set BEFORE any numpy/scipy/sklearn imports.
# See: https://github.com/scipy/scipy/issues/20294
#      https://github.com/scikit-learn/scikit-learn/pull/28692
# Only OPENBLAS needs to be pinned to 1; OMP can use multiple threads safely.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import time
import sys
import multiprocessing
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from afe import ArrowFieldEmbedding
from afe.reproducibility import (
    collect_machine_info,
    collect_package_versions,
    get_or_compute_spatial_embedding,
)

from benchmarks.config import (
    ALL_BACKENDS,
    ALL_MODES,
    sort_datasets_by_cost,
    get_optimal_arrow_count,
    get_arrow_counts,
)
from benchmarks.datasets import (
    DATASETS, LABELED_DATASETS,
    DATASET_CATEGORIES,
)
from benchmarks.io import (
    load_completed,
    load_all_results,
    append_result,
    _save_embedding,
    _metadata_for_row,
    _preprocessing_version,
)
from benchmarks.metrics import (
    compute_spatial_embedding,
    _compute_standard_metrics,
    _compute_afe_metrics,
)
from benchmarks.significance import (
    compute_significance,
    aggregate_significance,
)
from benchmarks.reporting import (
    print_summary,
    print_significance_summary,
    list_datasets,
)


# ── Utilities ───────────────────────────────────────────────────────────

def _effective_cpu_count():
    """Return the number of CPUs available to this process.

    Uses os.sched_getaffinity when available (containers), falling back
    to os.cpu_count().
    """
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


# ── Benchmark runners ───────────────────────────────────────────────────

def run_standalone(X, labels, backend_name, seed, spatial=None,
                   backend_params=None, save_embeddings=False,
                   dataset_name=None, package_versions=None,
                   machine_info=None, spatial_cache_hit=False,
                   spatial_cache_path=None, spatial_compute_seconds=None):
    """Run a standalone DR method (no AFE) as a baseline."""
    t0 = time.time()
    if spatial is None:
        spatial = compute_spatial_embedding(
            X, backend_name, seed, backend_params=backend_params
        )
        spatial_cache_hit = False
        spatial_cache_path = None
    Y = np.asarray(spatial, dtype=np.float32)
    elapsed = (
        float(spatial_compute_seconds)
        if spatial_compute_seconds is not None
        else time.time() - t0
    )

    if save_embeddings and dataset_name:
        _save_embedding(dataset_name, backend_name, None, 0, seed, X, Y)

    metrics = {
        "type": "benchmark",
        "method": f"{backend_name}_3d",
        "backend": backend_name,
        "afe_enhanced": False,
        "encoding_mode": None,
        "n_arrows": 0,
        "seed": seed,
        "time_seconds": elapsed,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "spatial_cache_hit": bool(spatial_cache_hit),
        "spatial_cache_path": str(spatial_cache_path) if spatial_cache_path else None,
        "spatial_compute_seconds": elapsed,
    }
    if dataset_name is not None:
        metrics.update(_metadata_for_row(
            dataset_name=dataset_name,
            backend_name=backend_name,
            backend_params=backend_params or {},
            seed=seed,
            encoding_mode=None,
            n_arrows=0,
            n_features=X.shape[1],
            package_versions=package_versions,
            machine_info=machine_info,
        ))

    standard_metrics, _ = _compute_standard_metrics(X, Y, labels=labels, seed=seed)
    metrics.update(standard_metrics)

    return metrics


def run_afe(X, labels, backend_name, encoding_mode, n_arrows, seed,
            spatial=None, backend_params=None, save_embeddings=False,
            dataset_name=None, package_versions=None, machine_info=None,
            spatial_cache_hit=False, spatial_cache_path=None,
            spatial_compute_seconds=None):
    """Run AFE with a specific backend, mode, and arrow count."""
    method_name = f"afe_{backend_name}_{encoding_mode}_{n_arrows}arr"

    if spatial is None:
        spatial = compute_spatial_embedding(
            X, backend_name, seed, backend_params=backend_params
        )
        spatial_cache_hit = False
        spatial_cache_path = None

    t0 = time.time()
    afe = ArrowFieldEmbedding(
        n_arrows=n_arrows,
        encoding_mode=encoding_mode,
        backend=np.asarray(spatial, dtype=np.float32),
        backend_kwargs=None,
        random_state=seed,
        normalize_arrows=False,
    )
    result = afe.fit_transform(X)
    elapsed = time.time() - t0

    spatial = result["spatial"]
    arrows = result["arrows"]

    if save_embeddings and dataset_name:
        _save_embedding(dataset_name, backend_name, encoding_mode, n_arrows,
                        seed, X, spatial, arrows)

    metrics = {
        "type": "benchmark",
        "method": method_name,
        "backend": backend_name,
        "afe_enhanced": True,
        "encoding_mode": encoding_mode,
        "n_arrows": n_arrows,
        "seed": seed,
        "time_seconds": elapsed,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "spatial_coordinates_shared": True,
        "spatial_cache_hit": bool(spatial_cache_hit),
        "spatial_cache_path": str(spatial_cache_path) if spatial_cache_path else None,
        "shared_spatial_compute_seconds": spatial_compute_seconds,
        "afe_encode_seconds": elapsed,
    }
    if dataset_name is not None:
        metrics.update(_metadata_for_row(
            dataset_name=dataset_name,
            backend_name=backend_name,
            backend_params=backend_params or {},
            seed=seed,
            encoding_mode=encoding_mode,
            n_arrows=n_arrows,
            n_features=X.shape[1],
            package_versions=package_versions,
            machine_info=machine_info,
        ))

    afe_metrics = _compute_afe_metrics(X, afe, spatial, arrows, labels=labels, seed=seed)
    metrics.update(afe_metrics)

    # Gap metadata
    gap = result["metadata"]["gap_report"]
    metrics["n_residual_dims"] = len(gap["residual_dims"])
    metrics["spatial_information_gap"] = gap["spatial_information_gap"]

    return metrics


# ── Orchestrator ────────────────────────────────────────────────────────

def compare_on_dataset(dataset_name, backends=None, arrow_counts=None,
                       n_seeds=3, modes=None, output_path=None,
                       completed=None, skip_standalone=False,
                       save_embeddings=False, spatial_cache_dir=None,
                       backend_params_by_name=None):
    """Run full comparison on one dataset across multiple backends and arrow counts.

    If arrow_counts is None, uses the optimal single count for the dataset's
    dimensionality. If arrow_counts is a list, sweeps over all positive counts.
    The zero-arrow comparison is the standalone spatial baseline, not an AFE run.
    """
    if backends is None:
        backends = ALL_BACKENDS
    if modes is None:
        modes = ALL_MODES
    if completed is None:
        completed = set()
    backend_params_by_name = backend_params_by_name or {}
    package_versions = collect_package_versions()
    machine_info = collect_machine_info()

    loader = DATASETS[dataset_name]
    try:
        X, y = loader()
    except (FileNotFoundError, ImportError) as e:
        print(f"\n  SKIPPING {dataset_name}: {e}")
        return []
    except (MemoryError, ValueError, RuntimeError) as e:
        print(f"\n  SKIPPING {dataset_name}: {e}")
        return []
    labels = y if dataset_name in LABELED_DATASETS else None
    n, d = X.shape

    # Arrow count selection
    if arrow_counts is not None:
        ds_arrow_counts = arrow_counts
    else:
        ds_arrow_counts = [get_optimal_arrow_count(d)]

    print(f"\n{'='*70}")
    print(f"  {dataset_name}  ({n} x {d})  arrows: {ds_arrow_counts}")
    print(f"{'='*70}")

    all_results = []
    total_run = 0
    total_skipped = 0

    failed_backends = set()  # Skip entire backend after first ImportError

    for backend_name in backends:
        if backend_name in failed_backends:
            continue

        backend_params = dict(backend_params_by_name.get(backend_name, {}))

        fixed_spatial_by_seed = {}
        spatial_cache_info_by_seed = {}
        for seed in range(n_seeds):
            try:
                spatial_t0 = time.time()
                spatial, cache_path, cache_hit = get_or_compute_spatial_embedding(
                    X=X,
                    dataset=dataset_name,
                    backend=backend_name,
                    seed=seed,
                    backend_params=backend_params,
                    preprocessing_version=_preprocessing_version(dataset_name),
                    cache_dir=spatial_cache_dir,
                    compute_fn=lambda b=backend_name, s=seed, p=backend_params: compute_spatial_embedding(
                        X, b, s, backend_params=p
                    ),
                )
                spatial_elapsed = time.time() - spatial_t0
                fixed_spatial_by_seed[seed] = spatial
                spatial_cache_info_by_seed[seed] = (
                    cache_path, cache_hit, spatial_elapsed
                )
            except ImportError as e:
                print(f"\n  {backend_name.upper()} NOT INSTALLED: {e}")
                print(f"    Skipping all {backend_name} runs for {dataset_name}")
                failed_backends.add(backend_name)
                break
            except (MemoryError, ValueError, RuntimeError) as e:
                print(f"\n  {backend_name.upper()} spatial computation FAILED: {e}")
                failed_backends.add(backend_name)
                break
            except Exception as e:
                print(f"\n  {backend_name.upper()} spatial computation FAILED: {e}")
                failed_backends.add(backend_name)
                break

        if backend_name in failed_backends:
            continue

        # Standalone baseline
        if not skip_standalone:
            print(f"\n  {backend_name.upper()} 3D (baseline)")
            for seed in range(n_seeds):
                method_name = f"{backend_name}_3d"
                key = (dataset_name, method_name, seed)
                if key in completed:
                    total_skipped += 1
                    continue

                print(f"    seed={seed} ...", end=" ", flush=True)
                try:
                    cache_path, cache_hit, spatial_elapsed = spatial_cache_info_by_seed[seed]
                    m = run_standalone(
                        X, labels, backend_name, seed,
                        spatial=fixed_spatial_by_seed[seed],
                        backend_params=backend_params,
                        save_embeddings=save_embeddings,
                        dataset_name=dataset_name,
                        package_versions=package_versions,
                        machine_info=machine_info,
                        spatial_cache_hit=cache_hit,
                        spatial_cache_path=cache_path,
                        spatial_compute_seconds=spatial_elapsed,
                    )
                    m["dataset"] = dataset_name
                    all_results.append(m)
                    if output_path:
                        append_result(output_path, m)
                    total_run += 1
                    print(f"done ({m['time_seconds']:.1f}s, "
                          f"kNN={m['knn_recall_k10']:.3f})")
                except ImportError as e:
                    print(f"NOT INSTALLED: {e}")
                    print(f"    Skipping all {backend_name} runs for {dataset_name}")
                    failed_backends.add(backend_name)
                    break
                except (MemoryError, ValueError, RuntimeError) as e:
                    print(f"FAILED: {e}")

        # AFE-enhanced variants
        if backend_name in failed_backends:
            continue

        for n_arrows in ds_arrow_counts:
            for mode in modes:
                if backend_name in failed_backends:
                    break
                print(f"\n  AFE+{backend_name} ({mode}, {n_arrows} arrows)")
                for seed in range(n_seeds):
                    method_name = f"afe_{backend_name}_{mode}_{n_arrows}arr"
                    key = (dataset_name, method_name, seed)
                    if key in completed:
                        total_skipped += 1
                        continue

                    print(f"    seed={seed} ...", end=" ", flush=True)
                    try:
                        cache_path, cache_hit, spatial_elapsed = spatial_cache_info_by_seed[seed]
                        m = run_afe(
                            X, labels, backend_name, mode, n_arrows, seed,
                            spatial=fixed_spatial_by_seed[seed],
                            backend_params=backend_params,
                            save_embeddings=save_embeddings,
                            dataset_name=dataset_name,
                            package_versions=package_versions,
                            machine_info=machine_info,
                            spatial_cache_hit=cache_hit,
                            spatial_cache_path=cache_path,
                            spatial_compute_seconds=spatial_elapsed,
                        )
                        m["dataset"] = dataset_name
                        all_results.append(m)
                        if output_path:
                            append_result(output_path, m)
                        total_run += 1
                        aknn = m.get("arrow_knn_recall_k10", 0)
                        print(f"done ({m['time_seconds']:.1f}s, "
                              f"kNN={m['knn_recall_k10']:.3f}, "
                              f"arrowkNN={aknn:.3f})")
                    except ImportError as e:
                        print(f"NOT INSTALLED: {e}")
                        print(f"    Skipping all {backend_name} runs for {dataset_name}")
                        failed_backends.add(backend_name)
                        break
                    except (MemoryError, ValueError, RuntimeError) as e:
                        print(f"FAILED: {e}")

    print(f"\n  [{dataset_name}] {total_run} runs, {total_skipped} skipped")
    return all_results


# ── Multiprocessing worker ──────────────────────────────────────────────

def _dataset_worker(task):
    """Run compare_on_dataset for a single dataset in a worker process.

    Each worker writes to its own temporary JSONL file to avoid
    cross-process file locking. The main process merges temp files
    after all workers finish.
    """
    ds = task["dataset_name"]
    worker_output = task["worker_output"]
    completed = set(tuple(k) for k in task.get("completed", []))
    try:
        results = compare_on_dataset(
            ds,
            backends=task["backends"],
            arrow_counts=task["arrow_counts"],
            n_seeds=task["n_seeds"],
            modes=task["modes"],
            output_path=worker_output,
            completed=completed,
            skip_standalone=task["skip_standalone"],
            save_embeddings=task["save_embeddings"],
            spatial_cache_dir=task["spatial_cache_dir"],
            backend_params_by_name=task.get("backend_params_by_name", {}),
        )
        return {"dataset": ds, "output": worker_output, "error": None}
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return {"dataset": ds, "output": worker_output, "error": err}


def _merge_worker_outputs(main_output, worker_outputs):
    """Append worker temp JSONL files into the main output file."""
    with open(main_output, "a") as out_f:
        for path in worker_outputs:
            if not Path(path).exists():
                continue
            with open(path) as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
            Path(path).unlink(missing_ok=True)


def _wait_for_procs(procs, min_remaining):
    """Wait until fewer than min_remaining processes are running.

    Returns a list of dataset names that finished during this wait.
    """
    import time
    finished_datasets = []
    while len(procs) >= min_remaining:
        finished = []
        for i, (ds, proc, log_file) in enumerate(procs):
            ret = proc.poll()
            if ret is not None:
                finished.append(i)
                finished_datasets.append(ds)
                log_file.close()
                if ret != 0:
                    print(f"\n  ERROR in {ds}: subprocess exited {ret}")
                    print(f"    Log: {log_file.name}")
                else:
                    print(f"  [{ds}] completed")
        for i in reversed(finished):
            procs.pop(i)
        if len(procs) >= min_remaining:
            time.sleep(0.5)
    return finished_datasets


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare DR methods: standalone vs AFE-enhanced"
    )
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, default=None,
                        choices=["general", "scrna", "all"],
                        help="Run all datasets in a category")
    parser.add_argument("--backends", type=str,
                        default=",".join(ALL_BACKENDS),
                        help=f"Comma-separated backends (default: all)")
    parser.add_argument("--backend-params", type=str, default=None,
                        help="JSON dict of backend-specific kwargs, e.g. "
                             "'{\"tsne\":{\"n_jobs\":-1},\"umap\":{\"n_neighbors\":15}}'")
    parser.add_argument("--n-arrows", type=str, default="optimal",
                        help="Arrow counts: 'optimal' (d-3, one per residual dim), "
                             "'sweep' (multi-count exploration), or comma-separated (e.g., 5,10,25)")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--modes", type=str, default="direct,pca,adaptive")
    parser.add_argument("--output", type=str, default="comparison_results.jsonl")
    parser.add_argument("--skip-standalone", action="store_true",
                        help="Skip standalone baseline runs")
    parser.add_argument("--skip-significance", action="store_true",
                        help="Skip significance testing")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Save embeddings to disk for post-hoc metric computation")
    parser.add_argument("--spatial-cache-dir", type=str,
                        default=str(Path(__file__).parent / "spatial_embeddings"),
                        help="Directory for fixed spatial coordinate cache")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel worker processes (default: 1, sequential). "
                             "Use -1 for all available CPU cores.")
    parser.add_argument("--list-datasets", action="store_true",
                        help="Print available datasets and exit")
    parser.add_argument("--precache", action="store_true",
                        help="Load all datasets sequentially to warm the cache, then exit")
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    # Resolve dataset list early for precache
    if args.category:
        if args.category == "all":
            datasets = list(DATASETS.keys())
        else:
            datasets = list(DATASET_CATEGORIES[args.category].keys())
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = ["swiss_roll", "hierarchical_gaussians"]

    if args.precache:
        print(f"Pre-caching {len(datasets)} datasets...")
        for i, ds in enumerate(datasets, 1):
            if ds not in DATASETS:
                print(f"  [{i}/{len(datasets)}] SKIPPING unknown dataset: {ds}")
                continue
            print(f"  [{i}/{len(datasets)}] Loading {ds} ...", end=" ", flush=True)
            try:
                loader = DATASETS[ds]
                X, y = loader()
                del X, y
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
        print("All datasets cached")
        return

    # Parse backend-specific parameters
    backend_params_by_name = {}
    if args.backend_params:
        try:
            backend_params_by_name = json.loads(args.backend_params)
        except json.JSONDecodeError as e:
            print(f"Error: --backend-params must be valid JSON. {e}")
            sys.exit(1)

    backends = [b.strip() for b in args.backends.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    arrow_counts = None  # None = optimal (d-3 per dataset)
    if args.n_arrows == "sweep":
        arrow_counts = "sweep"
    elif args.n_arrows not in ("optimal", "auto"):
        arrow_counts = [int(x.strip()) for x in args.n_arrows.split(",")]

    # Sort datasets slowest-first to prevent long-tail idle cores
    datasets = list(reversed(sort_datasets_by_cost(datasets)))

    # Resolve n_jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = _effective_cpu_count()
    n_jobs = max(1, n_jobs)

    # Load already-completed runs for resumability
    completed = load_completed(args.output)
    if completed:
        print(f"Resuming: {len(completed)} runs already completed in {args.output}")

    # Prepare worker tasks
    worker_dir = Path(args.output).parent / ".worker_tmp"
    worker_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    task_idx = 0
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Warning: unknown dataset '{ds}', skipping")
            continue
        if arrow_counts == "sweep":
            X_tmp, _ = DATASETS[ds]()
            ds_arrows = get_arrow_counts(X_tmp.shape[1])
            del X_tmp
        elif arrow_counts is None:
            ds_arrows = None
        else:
            ds_arrows = arrow_counts

        for backend_name in backends:
            worker_output = str(worker_dir / f"worker_{task_idx}_{ds}_{backend_name}.jsonl")
            tasks.append({
                "dataset_name": ds,
                "backends": [backend_name],
                "arrow_counts": ds_arrows,
                "n_seeds": args.n_seeds,
                "modes": modes,
                "worker_output": worker_output,
                "completed": [list(k) for k in completed],
                "skip_standalone": args.skip_standalone,
                "save_embeddings": args.save_embeddings,
                "spatial_cache_dir": args.spatial_cache_dir,
                "backend_params_by_name": backend_params_by_name,
            })
            task_idx += 1

    if n_jobs == 1:
        print(f"\nRunning benchmark sequentially on {len(tasks)} datasets...")
        for task in tasks:
            res = _dataset_worker(task)
            if res["error"]:
                print(f"\n  ERROR in {res['dataset']}: {res['error']}")
            _merge_worker_outputs(args.output, [res["output"]])
    else:
        print(f"\nRunning benchmark with {n_jobs} parallel workers on {len(tasks)} datasets...")
        # Launch independent subprocesses to avoid joblib's multiprocessing
        # detection (which forces n_jobs=1 in sklearn backends)
        import subprocess
        
        procs = []
        for task in tasks:
            ds = task["dataset_name"]
            backend_name = task["backends"][0]
            log_path = worker_dir / f"worker_{ds}_{backend_name}.log"
            log_file = open(log_path, "w")
            cmd = [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{Path(__file__).parent.parent}'); "
                f"from benchmarks.compare_methods import _dataset_worker; "
                f"import json; "
                f"task = json.loads({json.dumps(json.dumps(task))}); "
                f"res = _dataset_worker(task); "
                f"print(json.dumps(res))",
            ]
            env = os.environ.copy()
            env.pop('_MP_FORK_SERVER', None)  # Remove multiprocessing markers
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                text=True, env=env,
            )
            procs.append((ds, proc, log_file))
            if len(procs) >= n_jobs:
                # Wait for one to finish before starting more
                _wait_for_procs(procs, n_jobs)
        
        # Wait for remaining
        while len(procs) >= n_jobs:
            _wait_for_procs(procs, n_jobs)
        _wait_for_procs(procs, 1)

        # Collect results from worker output files
        worker_outputs = [task["worker_output"] for task in tasks]
        print(f"\nMerging worker outputs into {args.output}...")
        _merge_worker_outputs(args.output, worker_outputs)

    # Clean up worker temp dir
    if worker_dir.exists():
        try:
            worker_dir.rmdir()
        except OSError:
            pass

    # Load all results for summary/significance
    all_results = load_all_results(args.output)

    print_summary(all_results)

    # Significance testing
    if not args.skip_significance and len(all_results) > 0:
        print("\nComputing significance tests...")
        sig_results = compute_significance(all_results)
        aggregated = aggregate_significance(sig_results)

        for r in sig_results:
            append_result(args.output, r)

        print_significance_summary(sig_results, aggregated)

    total = len([r for r in all_results if r.get("type") != "significance"])
    print(f"\nBenchmark complete: {total} total runs in {args.output}")


if __name__ == "__main__":
    main()
