"""Prepare and verify immutable experiment datasets and their provenance."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from project_module.e_means.contracts import (
    ConditionConfig,
    DatasetManifest,
    PreparedDataset,
    derive_seed,
)

REVISION = "07b387891a221b7b073b83d2f752b76462e5fa03"
SOURCES = {
    "arxiv_nomic": (
        "arxiv-nomic-768-normalized.hdf5",
        4135431488,
        "8be0993b978b0d0ef023d21d878251a5ed09e058adb25994553c08388d37d414",
    ),
    "landmark_nomic": (
        "landmark-nomic-768-normalized.hdf5",
        2341733696,
        "9e870c21f3a33cea1adb9055ce0e7af0988557b5e4e21485a661f40365be286a",
    ),
}
ATTRIBUTION = {
    "arxiv_nomic": "VIBE by vector-index-bench, arXiv descriptive metadata (CC0), Nomic Embed Text v1.5 (Apache-2.0); subsets and group split. https://huggingface.co/datasets/vector-index-bench/vibe ; https://info.arxiv.org/help/api/tou.html ; https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 ; https://creativecommons.org/licenses/by/4.0/",
    "landmark_nomic": "VIBE by vector-index-bench, Google Landmarks index (CC0/public domain), Nomic Embed Vision v1.5 (Apache-2.0); subsets and group split. https://huggingface.co/datasets/vector-index-bench/vibe ; https://github.com/cvdfoundation/google-landmark#index-image-licenses ; https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5 ; https://creativecommons.org/licenses/by/4.0/",
}


_CACHE_NAMES = ("train", "pilot_train", "validation", "test")
_VERIFIED_CACHES: dict[Path, tuple[tuple[int, int, int], ...]] = {}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_vectors(x: npt.NDArray[np.float32], *, dimension: int, normalized: bool) -> None:
    """Validate finite C-contiguous float32 ``(rows, dimension)`` vectors."""
    if (
        not isinstance(x, np.ndarray)
        or x.ndim != 2
        or x.dtype != np.float32
        or not x.flags.c_contiguous
        or len(x) == 0
        or x.shape[1] != dimension
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid vector matrix or dimension")
    if normalized:
        for start in range(0, len(x), 8192):
            chunk = x[start : start + 8192]
            squared_norms = np.einsum("ij,ij->i", chunk, chunk)
            if np.any(squared_norms == 0):
                raise ValueError("normalized dataset contains zero norm")
            if not np.allclose(squared_norms, 1.0, rtol=1e-4, atol=1e-6):
                raise ValueError("normalized dataset must have unit norm")


def split_duplicate_groups(
    x: npt.NDArray[np.float32], *, validation_size: int, test_size: int, seed: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Split exact duplicate groups without overlap and return train/validation/test indices."""
    validate_vectors(x, dimension=x.shape[1], normalized=False)
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("holdout sizes must be positive")
    hashes = np.empty(len(x), dtype="V16")
    for i, row in enumerate(x):
        canonical = row.copy()
        canonical[canonical == 0] = 0
        hashes[i] = hashlib.blake2b(canonical.tobytes(), digest_size=16).digest()
    _, first, inverse, counts = np.unique(
        hashes, return_index=True, return_inverse=True, return_counts=True
    )
    # Verify equality within a hash group; a collision must never cause silent leakage.
    repeated_rows = np.flatnonzero(counts[inverse] > 1)
    for start in range(0, len(repeated_rows), 4096):
        rows = repeated_rows[start : start + 4096]
        if not np.array_equal(x[rows], x[first[inverse[rows]]]):
            raise ValueError("row hash collision")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    cursor = 0
    selections = []
    for target in (validation_size, test_size):
        start = cursor
        total = 0
        while total < target and cursor < len(order):
            total += int(counts[order[cursor]])
            cursor += 1
        if total < target or cursor >= len(order):
            raise ValueError("insufficient duplicate groups for holdouts and train")
        selections.append(order[start:cursor])
    labels = np.zeros(len(counts), dtype=np.uint8)
    labels[selections[0]] = 1
    labels[selections[1]] = 2
    return tuple(np.flatnonzero(labels[inverse] == label).astype(np.int64) for label in (0, 1, 2))  # type: ignore[return-value]


def _check_source_card(dataset_id: str) -> None:
    """Check the fixed-revision distribution card before preparing a public corpus."""
    url = f"https://huggingface.co/datasets/vector-index-bench/vibe/raw/{REVISION}/README.md"
    with urllib.request.urlopen(url, timeout=120) as response:
        card = response.read(1024 * 1024).decode("utf-8")
    name = SOURCES[dataset_id][0].removesuffix(".hdf5")
    count = {"arxiv_nomic": "1,344,643", "landmark_nomic": "760,757"}[dataset_id]
    matching = [line for line in card.splitlines() if name in line and line.startswith("|")]
    if (
        "license: cc-by-4.0" not in card
        or len(matching) != 1
        or count not in matching[0]
        or "| 768 |" not in matching[0]
    ):
        raise ValueError("source card metadata differs from fixed specification")


def _source(dataset_id: str, root: Path) -> Path:
    name, size, sha = SOURCES[dataset_id]
    path = root / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/vector-index-bench/vibe/resolve/{REVISION}/{name}"
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            if temporary.stat().st_size != size or _sha_file(temporary) != sha:
                raise ValueError("downloaded source size or SHA256 mismatch")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != size or _sha_file(path) != sha:
        raise ValueError("source size or SHA256 mismatch")
    return path


def _synthetic(
    condition: ConditionConfig, data_seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    params = condition.generator_params
    if set(params) - {"variant", "n_components", "validation_size", "test_size"}:
        raise ValueError("unknown synthetic parameter")
    variant = params.get("variant", "base")
    if not isinstance(variant, str) or variant not in {
        "base",
        "overlap",
        "aniso",
        "rare",
        "outlier",
        "duplicate",
        "interaction-1",
        "interaction-2",
    }:
        raise ValueError("unknown synthetic variant")

    def positive_parameter(key: str, default: int) -> int:
        value = params.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid synthetic parameter: {key}")
        return value

    components = positive_parameter("n_components", 256)
    n_val = positive_parameter("validation_size", 10000)
    n_test = positive_parameter("test_size", 10000)
    centers_rng = np.random.default_rng(derive_seed(data_seed, "data/centers"))
    noise_rng = np.random.default_rng(derive_seed(data_seed, "data/noise"))
    s = 0.5 if variant in {"overlap", "interaction-2"} else 3.0
    means = centers_rng.normal(0, s, (components, condition.dimension)).astype(np.float32)
    rare = variant in {"rare", "interaction-1"}
    if rare and components != 256:
        raise ValueError("rare variant requires 256 components")
    weights = np.full(components, 1 / components)
    if rare:
        means[:32] = centers_rng.normal(0, 10, (32, condition.dimension))
        weights[:32], weights[32:] = 0.01 / 32, 0.99 / 224
    scale = np.ones(condition.dimension, dtype=np.float32)
    if variant == "aniso":
        variance = np.linspace(0.1, 10, condition.dimension)
        scale = np.sqrt(variance / variance.mean()).astype(np.float32)
    arrays, groups = [], []
    for size in (condition.n_train, n_val, n_test):
        ids = noise_rng.choice(components, size=size, p=weights).astype(np.int64)
        x = np.empty((size, condition.dimension), dtype=np.float32)
        for start in range(0, size, 4096):
            end = min(size, start + 4096)
            noise = noise_rng.standard_normal((end - start, condition.dimension), dtype=np.float32)
            noise *= scale
            if rare:
                noise[ids[start:end] < 32] *= 0.5
            x[start:end] = means[ids[start:end]] + noise
        if variant == "outlier":
            outlier_indices = noise_rng.choice(size, size=size // 100, replace=False)
            x[outlier_indices] = noise_rng.normal(
                0, 20, (len(outlier_indices), condition.dimension)
            )
            ids[outlier_indices] = components
        arrays.append(x)
        groups.append(ids)
    if variant == "duplicate":
        x, ids = arrays[0], groups[0]
        candidates = noise_rng.permutation(len(x))
        first: dict[int, int] = {}
        replaced = 0
        for index in candidates:
            group = int(ids[index])
            if group in first and replaced < len(x) // 10:
                x[index] = x[first[group]]
                replaced += 1
            else:
                first.setdefault(group, int(index))
        if replaced < len(x) // 10:
            raise ValueError("insufficient same-component duplicate candidates")
    # Independent generation is overwhelmingly unlikely to duplicate across splits;
    # explicitly verify rather than assume it.
    seen: set[bytes] = set()
    for x in arrays:
        current = {hashlib.sha256(row.tobytes()).digest() for row in x}
        if seen.intersection(current):
            raise ValueError("synthetic duplicate or hash collision crosses splits")
        seen.update(current)
    return arrays, groups


def _artifact_dir(root: Path, dataset_id: str, condition_id: str, data_seed: int) -> Path:
    if any(
        not part or part in {".", ".."} or "/" in part or "\\" in part
        for part in (dataset_id, condition_id)
    ):
        raise ValueError("invalid dataset or condition id")
    return root / "processed" / dataset_id / condition_id / str(data_seed)


def _real_cache(condition: ConditionConfig, root: Path) -> Path:
    """Build one shared ordered pool and holdouts per public corpus."""
    cache = root / "processed" / condition.dataset_id / "_shared"
    marker = cache / "complete.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if (
            metadata.get("source_hash") != SOURCES[condition.dataset_id][2]
            or metadata.get("dimension") != condition.dimension
        ):
            raise ValueError("shared cache provenance mismatch")
        hashes = metadata.get("files")
        names = [f"{name}{suffix}.npy" for name in _CACHE_NAMES for suffix in ("", "_indices")]
        if not isinstance(hashes, dict) or set(hashes) != set(names):
            raise ValueError("shared cache file hashes missing")
        paths = [cache / name for name in names]
        stats = tuple(
            (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths
        )
        if _VERIFIED_CACHES.get(cache) != stats:
            if any(_sha_file(path) != hashes[path.name] for path in paths):
                raise ValueError("shared cache hash mismatch")
            _VERIFIED_CACHES[cache] = stats
        return cache
    _check_source_card(condition.dataset_id)
    source = _source(condition.dataset_id, root)
    cache.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as file:
        corpus = file["train"]
        expected_rows = {"arxiv_nomic": 1344643, "landmark_nomic": 760757}[condition.dataset_id]
        if (
            not isinstance(corpus, h5py.Dataset)
            or corpus.shape != (expected_rows, 768)
            or corpus.dtype != np.float32
        ):
            raise ValueError("source corpus shape or dtype mismatch")
        x = np.asarray(corpus)
    validate_vectors(x, dimension=condition.dimension, normalized=True)
    train, validation, test = split_duplicate_groups(
        x, validation_size=10000, test_size=10000, seed=42
    )
    main = np.random.default_rng(42).permutation(train)
    pilot = np.random.default_rng(derive_seed(42, "data/pilot-order")).permutation(train)
    for name, indices in (
        ("train", main),
        ("pilot_train", pilot),
        ("validation", validation),
        ("test", test),
    ):
        np.save(cache / f"{name}_indices.npy", indices, allow_pickle=False)
        # Save by chunks so the full corpus is never copied in memory again.
        out = np.lib.format.open_memmap(
            cache / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(indices), condition.dimension),
        )
        for start in range(0, len(indices), 4096):
            out[start : start + 4096] = x[indices[start : start + 4096]]
        out.flush()
        del out
    tmp = cache / "complete.json.tmp"
    tmp.write_text(
        json.dumps(
            {
                "source_hash": SOURCES[condition.dataset_id][2],
                "dimension": condition.dimension,
                "files": {
                    f"{name}{suffix}.npy": _sha_file(cache / f"{name}{suffix}.npy")
                    for name in _CACHE_NAMES
                    for suffix in ("", "_indices")
                },
            }
        )
    )
    os.replace(tmp, marker)
    return cache


def prepare_condition(condition: ConditionConfig, root: Path, data_seed: int) -> None:
    """Prepare one condition and seed, rejecting incompatible existing artifacts."""
    if data_seed not in condition.data_seeds:
        raise ValueError("data seed is not configured")
    dest = _artifact_dir(root, condition.dataset_id, condition.condition_id, data_seed)
    if (dest / "manifest.json").exists():
        manifest = DatasetManifest.model_validate_json((dest / "manifest.json").read_text())
        if (
            manifest.n_train != condition.n_train
            or manifest.dimension != condition.dimension
            or manifest.generator_params != condition.generator_params
        ):
            raise ValueError("existing manifest differs from condition")
        load_dataset(
            condition.dataset_id, root, condition_id=condition.condition_id, data_seed=data_seed
        )
        return
    dest.mkdir(parents=True, exist_ok=True)
    if condition.dataset_id == "synthetic":
        arrays, groups = _synthetic(condition, data_seed)
        source_hash = hashlib.sha256(
            json.dumps(condition.generator_params, sort_keys=True).encode()
        ).hexdigest()
        source_size, source_revision, license_name = 0, "synthetic-v1", "generated"
        split_seed, normalized = data_seed, False
        split_hash = hashlib.sha256(
            f"synthetic-v1:{data_seed}:{condition.condition_id}".encode()
        ).hexdigest()
    elif condition.dataset_id in SOURCES:
        cache = _real_cache(condition, root)
        pilot = condition.condition_id.startswith("pilot")
        order_name = "pilot_train" if pilot else "train"
        cached = [
            np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (order_name, "validation", "test")
        ]
        if condition.n_train > len(cached[0]):
            raise ValueError("requested n_train exceeds train pool")
        arrays = [cached[0][: condition.n_train], cached[1], cached[2]]
        indices = [
            np.load(cache / f"{name}_indices.npy", mmap_mode="r", allow_pickle=False)
            for name in (order_name, "validation", "test")
        ]
        split_hash = _digest_arrays([indices[0][: condition.n_train], indices[1], indices[2]])
        groups = []
        source_size, source_hash = SOURCES[condition.dataset_id][1:]
        source_revision, license_name, split_seed, normalized = REVISION, "CC BY 4.0", 42, True
    else:
        raise ValueError("unknown dataset id")
    for name, array in zip(("train", "validation", "test"), arrays, strict=True):
        validate_vectors(array, dimension=condition.dimension, normalized=normalized)
        if normalized:
            cached_name = (
                "pilot_train"
                if name == "train" and condition.condition_id.startswith("pilot")
                else name
            )
            target = dest / f"{name}.npy"
            target.unlink(missing_ok=True)
            os.link(cache / f"{cached_name}.npy", target)
            index_target = dest / f"{name}_indices.npy"
            index_target.unlink(missing_ok=True)
            os.link(cache / f"{cached_name}_indices.npy", index_target)
        else:
            np.save(dest / f"{name}.npy", array, allow_pickle=False)
    for name, group in zip(("train", "validation", "test"), groups, strict=False):
        np.save(dest / f"{name}_groups.npy", group, allow_pickle=False)
    manifest = DatasetManifest(
        dataset_id=condition.dataset_id,
        condition_id=condition.condition_id,
        data_seed=data_seed,
        source_revision=source_revision,
        source_hash=source_hash,
        source_size_bytes=source_size,
        data_hash=_digest_arrays(arrays),
        group_hash=_digest_arrays(groups) if groups else None,
        split_hash=split_hash,
        split_seed=split_seed,
        split_algorithm="exact-row-groups-v1" if normalized else "independent-streams-v1",
        dimension=condition.dimension,
        n_train=len(arrays[0]),
        n_validation=len(arrays[1]),
        n_test=len(arrays[2]),
        normalized=normalized,
        generator_params=condition.generator_params,
        license=license_name,
        attribution=ATTRIBUTION[condition.dataset_id]
        if normalized
        else "Generated Gaussian mixture",
    )
    temporary = dest / "manifest.json.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    os.replace(temporary, dest / "manifest.json")


def load_dataset(
    dataset_id: str, root: Path, *, condition_id: str, data_seed: int
) -> PreparedDataset:
    """Load verified read-only float32 matrices and optional aligned int64 groups."""
    dest = _artifact_dir(root, dataset_id, condition_id, data_seed)
    path = dest / "manifest.json"
    manifest = DatasetManifest.model_validate_json(path.read_text())
    if (manifest.dataset_id, manifest.condition_id, manifest.data_seed) != (
        dataset_id,
        condition_id,
        data_seed,
    ):
        raise ValueError("manifest identity mismatch")
    names = ("train", "validation", "test")
    if dataset_id == "synthetic":
        if (
            manifest.source_revision != "synthetic-v1"
            or manifest.split_seed != data_seed
            or manifest.normalized
        ):
            raise ValueError("synthetic manifest provenance mismatch")
        split_hash = hashlib.sha256(f"synthetic-v1:{data_seed}:{condition_id}".encode()).hexdigest()
    elif dataset_id in SOURCES:
        if (
            manifest.source_revision != REVISION
            or manifest.source_hash != SOURCES[dataset_id][2]
            or manifest.source_size_bytes != SOURCES[dataset_id][1]
            or manifest.split_seed != 42
            or not manifest.normalized
        ):
            raise ValueError("source manifest provenance mismatch")
        indices = [
            np.load(dest / f"{name}_indices.npy", mmap_mode="r", allow_pickle=False)
            for name in names
        ]
        split_hash = _digest_arrays([indices[0][: manifest.n_train], indices[1], indices[2]])
    else:
        raise ValueError("unknown dataset id")
    if split_hash != manifest.split_hash:
        raise ValueError("split hash mismatch")
    arrays = [np.load(dest / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names]
    arrays[0] = arrays[0][: manifest.n_train]
    for array, expected in zip(
        arrays, (manifest.n_train, manifest.n_validation, manifest.n_test), strict=True
    ):
        validate_vectors(array, dimension=manifest.dimension, normalized=manifest.normalized)
        if len(array) != expected:
            raise ValueError("manifest row count mismatch")
    if _digest_arrays(arrays) != manifest.data_hash:
        raise ValueError("dataset hash mismatch")
    groups = (
        [np.load(dest / f"{name}_groups.npy", mmap_mode="r", allow_pickle=False) for name in names]
        if manifest.group_hash is not None
        else [None, None, None]
    )
    if manifest.group_hash is not None and _digest_arrays(groups) != manifest.group_hash:  # type: ignore[arg-type]
        raise ValueError("group hash mismatch")
    return PreparedDataset(
        train=arrays[0],
        validation=arrays[1],
        test=arrays[2],
        train_groups=groups[0],
        validation_groups=groups[1],
        test_groups=groups[2],
        n_train=manifest.n_train,
        split_hash=manifest.split_hash,
        data_hash=manifest.data_hash,
        manifest_path=path,
    )
