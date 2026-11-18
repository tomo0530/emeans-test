"""Export deterministic scalar parameters; never load vectors, train, or import Torch."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import faiss
import yaml
from project_module.e_means.contracts import ExperimentConfig, Method, RunSpec, derive_seed
from project_module.e_means.methods import emeans_params
from sklearn.cluster import KMeans, MiniBatchKMeans

W = Path(".superpowers/sdd/2026-11-17-e-means-implementation-plan")
assert sys.version_info[:2] == (3, 14)
assert importlib.metadata.version("faiss-cpu") == "1.16.0"
wheel_path = importlib.metadata.distribution("faiss-cpu").locate_file("faiss/contrib/e_means")
if not isinstance(wheel_path, Path):
    raise TypeError("The frozen faiss-cpu wheel must be installed on the filesystem")
wheel: Path = wheel_path
spec = importlib.util.spec_from_file_location("_frozen_emeans_base", wheel / "base.py")
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
assert "torch" not in sys.modules, "Scalar export must not import Torch or initialize CUDA"

CP_FIELDS = (
    "afkmc2_chain_length",
    "check_input_data_for_NaNs",
    "decode_block_size",
    "early_stop_threshold",
    "frozen_centroids",
    "init_method",
    "int_centroids",
    "max_points_per_centroid",
    "min_points_per_centroid",
    "niter",
    "nredo",
    "seed",
    "spherical",
    "update_index",
    "use_faster_subsampling",
    "use_super_kmeans",
    "verbose",
)


def encoded(value: object) -> bytes:
    """Encode a value as canonical, finite JSON bytes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(method: Method, n: int, d: int, k: int, budget: float, seed: int) -> dict[str, Any]:
    """Build the scalar prediction using installed constructor defaults and wheel helpers."""
    method_seed = derive_seed(seed, "method/" + method) if method != "torch_lloyd" else None
    if method in ["numpy_e_means", "torch_e_means"]:
        run_spec = RunSpec(
            run_id="scalar-export",
            dataset_id="scalar-export",
            condition_id="scalar-export",
            method=method,
            device="cpu" if method == "numpy_e_means" else "cuda",
            measurement_id="scalar-export",
            n_train=n,
            k=k,
            seed=seed,
            data_seed=seed,
            threads=1,
            timing_repeat=0,
            budget=budget,
            params={},
        )
        requested = emeans_params(run_spec)
        chunk = (
            65536
            if method == "numpy_e_means"
            else min(2**20, max(8192, 2**26 // max(d, min(k, 4096))))
        )
        law, reason = base._resolve_batch_law(requested["batch_law"], d)
        epochs = int(budget)
        batch, alpha = base._resolve_recipe(
            n, k, epochs, law, requested["batch_size"], requested["alpha0"]
        )
        epoch_rows = base._epoch_rows(n, epochs)
        epoch_steps = [(rows + batch - 1) // batch for rows in epoch_rows]
        recipe = {
            "name": "canonical_v1",
            "batch_law": law,
            "batch_law_requested": requested["batch_law"],
            "batch_law_reason": reason,
            "batch_size": batch,
            "batch_size_requested": requested["batch_size"],
            "alpha0": alpha,
            "alpha0_source": "step_law",
            "nepochs": epochs,
            "steps_per_pass": (n + batch - 1) // batch,
            "total_steps": sum(epoch_steps),
            "epoch_rows": epoch_rows,
            "epoch_steps": epoch_steps,
            "tail_policy": "keep",
            "alpha_min_factor": base._ALPHA_MIN_FACTOR,
            "check_unused_every": base._CHECK_UNUSED_EVERY,
            "unused_threshold": base._resolve_unused_threshold(requested["unused_threshold"], d),
            "split_epsilon": base._SPLIT_EPSILON,
            "init_usage": base._INIT_USAGE,
            "seed": method_seed,
            "backend": "faiss" if method == "numpy_e_means" else "torch",
            "assignment_chunk_size": chunk,
        }
        params = {
            **requested,
            "max_points_per_centroid": math.ceil(n / k),
            "capture_assignments": False,
            "spherical": False,
        }
        if method == "torch_e_means":
            params.update(quantizer=None, assignment_backend="native", centroid_chunk_size=4096)
            recipe.update(
                device="cuda:0",
                representation="float32",
                assignment_backend="native",
                assignment_backend_requested="native",
                centroid_chunk_size=4096,
            )
        schedule = [
            base._scheduled_alpha(recipe["alpha0"], i, recipe["total_steps"])
            for i in range(recipe["total_steps"])
        ]
        return {
            "recorded_params": params,
            "recipe": recipe,
            "expected_processed_rows": n * int(budget),
            "schedule": {
                "applied_sha256": hashlib.sha256(encoded(schedule)).hexdigest(),
                "first": schedule[0],
                "last_applied": schedule[-1],
                "endpoint_not_applied": base._scheduled_alpha(
                    recipe["alpha0"], recipe["total_steps"], recipe["total_steps"]
                ),
            },
        }
    if method == "faiss_lloyd":
        model = faiss.Kmeans(
            d,
            k,
            niter=int(budget),
            nredo=1,
            seed=method_seed,
            max_points_per_centroid=math.ceil(n / k),
            min_points_per_centroid=1,
            spherical=False,
        )
        return {
            "recorded_params": {key: getattr(model.cp, key) for key in CP_FIELDS},
            "maximum_processed_rows": n * int(budget),
        }
    if method == "mini_batch":
        batch = min(n, max(4096, k))
        steps = math.ceil(budget * n / batch)
        model = MiniBatchKMeans(
            n_clusters=k,
            init="random",
            n_init=1,
            batch_size=batch,
            compute_labels=False,
            random_state=method_seed,
            tol=0.0,
            max_no_improvement=None,
        )
        params = model.get_params()
        params["init"] = "supplied"
        return {
            "recorded_params": params,
            "planned_batches": steps,
            "expected_processed_rows": steps * batch,
            "sample_with_replacement": True,
        }
    if method == "elkan":
        model = KMeans(
            n_clusters=k,
            init="random",
            n_init=1,
            max_iter=int(budget),
            tol=0.0,
            algorithm="elkan",
            random_state=method_seed,
        )
        params = model.get_params()
        params["init"] = "supplied"
        return {
            "recorded_params": params,
            "maximum_processed_rows": n * int(budget),
            "stop_condition": "strict assignment convergence or iteration budget",
        }
    if method != "torch_lloyd":
        raise ValueError(f"unsupported method: {method}")
    return {
        "assignment_backend": "native",
        "assignment_chunk_size": 4096,
        "centroid_chunk_size": 4096,
        "empty_center_policy": "retain",
        "planned_iterations": int(budget),
        "expected_processed_rows": n * int(budget),
    }


def build_records(configs: dict[str, ExperimentConfig]) -> dict[Method, list[dict[str, Any]]]:
    """Build seed-independent records for each method and condition."""
    methods = [m.method for m in configs["core"].methods]
    records: dict[Method, list[dict[str, Any]]] = {method: [] for method in methods}
    for name, config in configs.items():
        assert config.seeds == list(range(10)) and config.phase == "main"
        for cond in config.conditions:
            for method in config.methods:
                assert method.params == {}, (
                    "Exporter only covers this fixed default-parameter matrix"
                )
                for budget in method.budgets:
                    params = build(method.method, cond.n_train, cond.dimension, cond.k, budget, 0)
                    seed_field = (
                        "seed"
                        if method.method in ["faiss_lloyd", "numpy_e_means", "torch_e_means"]
                        else ("random_state" if method.method in ["mini_batch", "elkan"] else None)
                    )
                    if seed_field:
                        target = (
                            params["recipe"]
                            if method.method.endswith("e_means")
                            else params["recorded_params"]
                        )
                        target.pop(seed_field)
                    records[method.method].append(
                        {
                            "config": name,
                            "condition_id": cond.condition_id,
                            "dataset_id": cond.dataset_id,
                            "n_train": cond.n_train,
                            "dimension": cond.dimension,
                            "k": cond.k,
                            "budget": budget,
                            "requested_config_params": {},
                            "seed_field": seed_field,
                            "method_seed_table_ref": method.method if seed_field else None,
                            "effective_seed_independent": params,
                        }
                    )
    assert sum(len(rows) for rows in records.values()) == 880
    return records


def compare_pilots(input_roots: list[Path]) -> list[dict[str, Any]]:
    """Compare predictions with each successful primary pilot attempt."""
    comparisons = []
    for input_root in input_roots:
        for ledger in (
            [input_root / "jobs.json"]
            if (input_root / "jobs.json").exists()
            else sorted(input_root.rglob("jobs.json"))
        ):
            root = ledger.parent
            config = json.loads((root / "config.snapshot.json").read_text())
            conditions = {c["condition_id"]: c for c in config["conditions"]}
            assert config["phase"] == "pilot" and not config["measurement"]["evaluate_test"]
            for job in json.loads(ledger.read_text()):
                path = None
                for attempt in sorted((root / job["run_id"]).glob("attempt-*")):
                    status_path = attempt / "status.json"
                    if not status_path.exists():
                        break
                    status = json.loads(status_path.read_text())["status"]
                    if status == "interrupted":
                        continue
                    if status == "success":
                        path = attempt
                    break
                if path is None:
                    continue
                m = json.loads((path / "metrics.json").read_text())
                assert m["test"] is None and m["phase"] == "pilot"
                c = conditions[job["condition_id"]]
                key = (
                    job["method"],
                    job["n_train"],
                    c["dimension"],
                    job["k"],
                    job["budget"],
                    job["seed"],
                )
                expected = build(
                    job["method"],
                    job["n_train"],
                    c["dimension"],
                    job["k"],
                    job["budget"],
                    job["seed"],
                )
                for label in ["recipe", "recorded_params"]:
                    if label not in expected:
                        continue
                    actual = m["stats"]["recipe" if label == "recipe" else "resolved_params"]
                    for field, value in expected[label].items():
                        assert actual[field] == value, (key, label, field, value, actual.get(field))
                if job["method"] == "torch_lloyd":
                    for field in [
                        "assignment_backend",
                        "assignment_chunk_size",
                        "centroid_chunk_size",
                        "empty_center_policy",
                    ]:
                        assert m["stats"][field] == expected[field], (key, field)
                    assert m["stats"]["actual_iterations"] == expected["planned_iterations"]
                if "expected_processed_rows" in expected:
                    assert m["stats"]["processed_rows"] == expected["expected_processed_rows"], key
                elif "maximum_processed_rows" in expected:
                    assert 0 < m["stats"]["processed_rows"] <= expected["maximum_processed_rows"], (
                        key
                    )
                comparisons.append(
                    {
                        "key": key,
                        "condition_id": job["condition_id"],
                        "input_root": input_root.name,
                        "attempt": str(path.relative_to(input_root)),
                        "match": True,
                    }
                )
    return comparisons


def main() -> None:
    """Export the parameter snapshot selected by command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--kind", choices=["draft", "pre-test-candidate"], default="draft")
    parser.add_argument("--compare-pilot-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    configs = {
        name: ExperimentConfig.model_validate(
            yaml.safe_load(Path("configs/experiments/e_means", name + ".yaml").read_text())
        )
        for name in ["core", "stress"]
    }
    if args.kind == "pre-test-candidate":
        assert (
            json.loads((W / "pre-main-campaign-status.json").read_text())["status"]
            == "ready-for-protocol-gate"
        )
        cohort = json.loads((W / "profile-qualification.json").read_text())["selected_main_cohort"]
        assert all(c.measurement.cohort == cohort for c in configs.values())
    methods = [method.method for method in configs["core"].methods]
    records = build_records(configs)
    comparisons = compare_pilots(args.compare_pilot_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    files = {}
    seeds = {
        "method_seeds": {
            method: {str(seed): derive_seed(seed, "method/" + method) for seed in range(10)}
            for method in methods
            if method != "torch_lloyd"
        },
        "epoch_seed_rule": "(method_seed*1000003+epoch)%2**32; zero-based epoch",
        "epoch_seeds": {
            method: {
                str(seed): [
                    base._epoch_seed(derive_seed(seed, "method/" + method), epoch)
                    for epoch in range(12)
                ]
                for seed in range(10)
            }
            for method in ["numpy_e_means", "torch_e_means"]
        },
    }
    values = {"seeds.json": seeds}
    for config_name in configs:
        for method, rows in records.items():
            values[config_name + "/" + method + ".json"] = {
                "method": method,
                "config": config_name,
                "rows": [row for row in rows if row["config"] == config_name],
            }
    for name, value in values.items():
        path = args.output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(value) + b"\n")
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    source_paths = list(Path("project_module/src/project_module/e_means/methods").glob("*.py")) + [
        Path("project_module/src/project_module/e_means/contracts.py"),
        Path("project_module/src/project_module/e_means/runner.py"),
        Path("uv.lock"),
    ]
    index = {
        "schema_version": 1,
        "kind": args.kind,
        "approval": "separate main-protocol gate required",
        "generator_sha256": digest(Path(__file__)),
        "git_head_at_export": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["faiss-cpu", "torch", "numpy", "scikit-learn", "pydantic"]
        },
        "config_sha256": {
            name: digest(Path("configs/experiments/e_means", name + ".yaml")) for name in configs
        },
        "cohorts": {name: c.measurement.cohort for name, c in configs.items()},
        "resources": {name: c.resources.model_dump(mode="json") for name, c in configs.items()},
        "gpu_runtime_request": {
            "deterministic_algorithms": True,
            "warn_only": False,
            "cublas_workspace_config": ":4096:8",
            "graph_policy": "upstream_auto",
            "assignment_backend": "native",
            "dtype": "float32",
            "quantizer": None,
        },
        "field_interpretation": {
            "recorded_params": "Exact adapter-recorded fields; e-means None/auto values are requests, with effective values in recipe",
            "seed_field": "Omitted from common parameters; merge the exact value from seeds.json method_seeds[method][experiment_seed]",
            "schedule": "SHA256 of logical upstream Python scalar alphas before backend numeric casts; not an observed CUDA trace; first/last update differ from the unused endpoint",
            "initialization": "Supplied exact-unique train-row centers; actual matrix hashes are frozen separately in the run ledger",
        },
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "wheel_source_sha256": {
            name: digest(wheel / name)
            for name in [
                "base.py",
                "e_means.py",
                "e_means_torch.py",
                "e_means_quantized.py",
                "torch_backend.py",
            ]
        },
        "seed_independent_records": 880,
        "independent_seed_jobs_before_timing_repeats": 8800,
        "files": files,
        "torch_imported": "torch" in sys.modules,
        "pilot_metadata_comparisons": comparisons,
        "scope": "Scalar prediction and constructor defaults only; supplied init matrices remain bound by the frozen run ledger",
        "not_predicted": [
            "actual_iterations",
            "actual_revival/reassignment",
            "actual_graph_replays",
            "loss",
            "time",
            "memory",
        ],
        "implementation_difference": "Installed base.py uses init_usage2**-16; its source notes shipped fit coefficients differ from paper Table2; freeze wheel rules, not paper pseudocode",
    }
    assert not index["torch_imported"]
    (args.output_root / "index.json").write_bytes(encoded(index) + b"\n")
    print(
        json.dumps(
            {
                "root": str(args.output_root),
                "kind": args.kind,
                "records": 880,
                "pilot_metadata_comparisons": len(comparisons),
                "files": files,
                "torch_imported": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
