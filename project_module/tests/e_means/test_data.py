"""Prepared dataset and split invariants."""

from pathlib import Path

import numpy as np
import pytest
from project_module.e_means.contracts import ConditionConfig
from project_module.e_means.data import (
    load_dataset,
    prepare_condition,
    split_duplicate_groups,
    validate_vectors,
)


def _condition(seed: int = 0, condition_id: str = "S0") -> ConditionConfig:
    return ConditionConfig(
        condition_id=condition_id,
        dataset_id="synthetic",
        n_train=100,
        k=4,
        dimension=4,
        data_seeds=[seed],
        generator_params={
            "n_components": 4,
            "validation_size": 30,
            "test_size": 30,
            "variant": "duplicate",
        },
    )


def test_duplicate_groups_never_cross_splits() -> None:
    x = np.repeat(np.arange(80, dtype=np.float32).reshape(40, 2), 2, axis=0)
    split = split_duplicate_groups(x, validation_size=12, test_size=12, seed=42)
    again = split_duplicate_groups(x, validation_size=12, test_size=12, seed=42)
    assert all(np.array_equal(a, b) for a, b in zip(split, again, strict=True))
    sets = [set(map(tuple, x[index])) for index in split]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ],
)
def test_rejects_invalid_normalized_dataset(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_vectors(values, dimension=2, normalized=True)


def test_condition_and_data_seed_select_exact_manifest(tmp_path: Path) -> None:
    prepare_condition(_condition(), tmp_path, 0)
    prepare_condition(_condition(1, "pilot-S0"), tmp_path, 1)
    main = load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)
    pilot = load_dataset("synthetic", tmp_path, condition_id="pilot-S0", data_seed=1)
    assert not np.array_equal(main.train, pilot.train)
    with pytest.raises((FileNotFoundError, ValueError)):
        load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=1)


def test_synthetic_group_ids_survive_round_trip(tmp_path: Path) -> None:
    prepare_condition(_condition(), tmp_path, 0)
    result = load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)
    assert result.train_groups is not None and len(result.train_groups) == 100
    assert set(result.train_groups) <= {0, 1, 2, 3}
    assert not result.train.flags.writeable


def test_allows_origin_for_general_l2_input() -> None:
    validate_vectors(np.array([[0.0, 0.0]], dtype=np.float32), dimension=2, normalized=False)


def test_rejects_tampered_split_hash(tmp_path: Path) -> None:
    import json

    prepare_condition(_condition(), tmp_path, 0)
    path = tmp_path / "processed/synthetic/S0/0/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split_hash"] = "corrupt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="split hash"):
        load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)


def test_rejects_tampered_data(tmp_path: Path) -> None:
    prepare_condition(_condition(), tmp_path, 0)
    path = tmp_path / "processed/synthetic/S0/0/train.npy"
    array = np.load(path, mmap_mode="r+")
    array[0, 0] += 1
    array.flush()
    with pytest.raises(ValueError, match="dataset hash"):
        load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)


def test_duplicate_condition_replaces_ten_percent_of_train_rows(tmp_path: Path) -> None:
    prepare_condition(_condition(), tmp_path, 0)
    result = load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)
    assert len(np.unique(result.train, axis=0)) == 90
    train = set(map(tuple, result.train))
    assert train.isdisjoint(map(tuple, result.validation))
    assert train.isdisjoint(map(tuple, result.test))


def test_rejects_unknown_synthetic_variant(tmp_path: Path) -> None:
    condition = _condition().model_copy(update={"generator_params": {"variant": "typo"}})
    with pytest.raises(ValueError, match="variant"):
        prepare_condition(condition, tmp_path, 0)


def test_outlier_fraction_is_exactly_one_percent(tmp_path: Path) -> None:
    condition = _condition().model_copy(
        update={
            "generator_params": {
                "n_components": 4,
                "validation_size": 100,
                "test_size": 100,
                "variant": "outlier",
            }
        }
    )
    prepare_condition(condition, tmp_path, 0)
    result = load_dataset("synthetic", tmp_path, condition_id="S0", data_seed=0)
    assert np.count_nonzero(result.train_groups == 4) == 1
    assert np.count_nonzero(result.validation_groups == 4) == 1
    assert np.count_nonzero(result.test_groups == 4) == 1


def test_rejects_non_unit_distribution_declared_normalized() -> None:
    with pytest.raises(ValueError, match="unit norm"):
        validate_vectors(np.array([[1.0, 1.0]], dtype=np.float32), dimension=2, normalized=True)
