"""Small end-to-end numerical checks for runner artifacts."""

import json
from pathlib import Path

import numpy as np
from project_module.e_means.contracts import ExperimentConfig
from project_module.e_means.data import load_dataset, prepare_condition
from project_module.e_means.runner import run_phase


def test_smoke_uses_same_init_and_independent_final_mse(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "plan_revision": "test",
            "phase": "pilot",
            "conditions": [
                {
                    "condition_id": "smoke",
                    "dataset_id": "synthetic",
                    "n_train": 40,
                    "k": 2,
                    "dimension": 3,
                    "data_seeds": [7],
                    "generator_params": {
                        "n_components": 2,
                        "validation_size": 8,
                        "test_size": 8,
                        "variant": "base",
                    },
                }
            ],
            "methods": [
                {"method": "faiss_lloyd", "budgets": [1], "params": {}},
                {"method": "mini_batch", "budgets": [1], "params": {}},
            ],
            "seeds": [7],
            "measurement": {
                "cohort": "smoke",
                "repeats": 1,
                "evaluate_test": False,
                "repeat_rules": [],
            },
            "resources": {
                "cpu_threads": 1,
                "cpu_affinity": [],
                "gpu_pool": [0],
                "gpu_memory_limit_bytes": 25769803776,
                "host_memory_limit_bytes": 412316860416,
            },
        }
    )
    data = tmp_path / "data"
    output = tmp_path / "output"
    prepare_condition(config.conditions[0], data, 7)
    jobs = run_phase(config, data, output, phase="cpu", resume=True)
    dataset = load_dataset("synthetic", data, condition_id="smoke", data_seed=7)
    hashes = set()
    cache_hits = []
    for job in jobs:
        attempt = output / job.run_id / "attempt-0001"
        assert json.loads((attempt / "status.json").read_text())["status"] == "success"
        metrics = json.loads((attempt / "metrics.json").read_text())
        assert metrics["timing"]["metadata"]["e2e_includes_dataset_loading"] is False
        assert metrics["timing"]["metadata"]["data_load_seconds"] >= 0
        cache_hits.append(metrics["timing"]["metadata"]["init_cache_hit"])
        centers = np.load(attempt / "centroids.npy")
        points = dataset.train.astype(np.float64)
        centers64 = centers.astype(np.float64)
        reference = np.min(
            np.sum((points[:, None, :] - centers64[None, :, :]) ** 2, axis=2), axis=1
        ).mean()
        assert np.isclose(metrics["train"]["mse"], reference, rtol=1e-5, atol=1e-6)
        hashes.add(metrics["init_hash"])
    assert len(hashes) == 1
    assert cache_hits == [False, True]
