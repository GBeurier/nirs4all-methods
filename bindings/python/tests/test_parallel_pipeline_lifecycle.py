# SPDX-License-Identifier: CECILL-2.1
"""Real concurrent CDLL calls; independent native state over immutable inputs."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from _native_fit_witness import witness

from n4m import MultimodalPipeline, N4MError, build_capabilities
from n4m._ffi import lib
from n4m.roles import RolePipeline


def shared_case():
    rng = np.random.default_rng(831)
    n = 4096
    blocks = {
        "nir": rng.normal(size=(n, 128)),
        "image": rng.normal(size=(n, 2, 3, 2)),
        "series": rng.normal(size=(n, 4, 2)),
        "metadata": np.column_stack((np.arange(n) / n, ["A", "é", "猫", "A"] * (n // 4))).astype(object),
    }
    y = 0.4 * blocks["nir"][:, 0] - blocks["nir"][:, 1] + rng.normal(size=n) / 10
    representations = ("signal_1d", "rgb_image", "series_mv", "tabular_mixed")
    schemas = {name: {"representation_id": representation,
                       "input_shape": list(value.shape[1:]), "dtype": str(value.dtype),
                       "identity": "parallel-lifecycle:" + name}
               for (name, value), representation in zip(blocks.items(), representations, strict=True)}
    recipe = {
        "schema_version": 1, "fusion": "early", "source_order": list(blocks),
        "encoders": {
            "nir": {"kind":"standard_scaler", "with_mean":True, "with_std":True},
            "image": {"kind":"tensor_pca", "n_components":3, "whiten":False, "random_state":42},
            "series": {"kind":"tensor_pca", "n_components":2, "whiten":False, "random_state":31},
            "metadata": {"kind":"column_transformer", "numeric_columns":[0],
                         "categorical_columns":[1], "with_mean":True, "with_std":True,
                         "handle_unknown":"ignore", "sparse_output":False, "drop":None},
        },
        "source_weights": {name:1.0 for name in blocks},
        "model": {"method_id":"models.regularized.ridge",
                  "params":{"alpha":0.3, "center_x":True, "center_y":True, "scale_x":False}},
    }
    for array in [*blocks.values(), y]:
        array.flags.writeable = False
    return blocks, y, recipe, schemas


def input_hashes(blocks, y):
    return {name: hashlib.sha256(
        json.dumps(value.tolist(), ensure_ascii=False).encode()
        if value.dtype.kind in "OUS" else value.tobytes()
    ).hexdigest() for name, value in {**blocks, "target":y}.items()}


def peak_rss():
    try:
        import resource
    except ImportError:
        return None
    # ru_maxrss is KiB on Linux, bytes on macOS; retain native units honestly.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.parametrize("workers", [2, 3, 4])
@pytest.mark.parametrize("kind", ["role", "multimodal"])
@pytest.mark.parametrize("serialized", [False, True])
def test_native_fit_probe_control_replay_close_and_unchanged_shared_inputs(
    workers, kind, serialized, monkeypatch, record_property
):
    assert build_capabilities().sequential_cpu, "Witness requires the sequential numerical profile"
    blocks, y, recipe, schemas = shared_case()
    before = input_hashes(blocks, y)
    steps = ["models.regularized.ridge"]
    inputs = blocks["nir"] if kind == "role" else blocks
    # Direct construction avoids from_steps' transient validation allocation.
    make = (lambda: RolePipeline(steps)) if kind == "role" else (
        lambda: MultimodalPipeline(recipe, schemas)
    )
    serial = make()
    try:
        serial.fit(inputs, y)
        expected = serial.predict(inputs)
    finally:
        serial.close()
    ready_close = threading.Barrier(workers, timeout=30)
    lock = threading.Lock()
    owners, destroyed = [], []
    tracked_lifetimes = set()
    destroy_symbol = "n4m_" + ("role" if kind == "role" else "multimodal") + "_pipeline_destroy"
    native_destroy = getattr(lib, destroy_symbol)

    def observed_destroy(handle):
        if handle is not None and handle.value:
            with lock:
                lifetime = (threading.get_ident(), handle.value)
                if lifetime in tracked_lifetimes:
                    destroyed.append(lifetime)
        return native_destroy(handle)

    def worker():
        model, restored = make(), None
        try:
            model.fit(inputs, y)
            np.testing.assert_array_equal(model.predict(inputs), expected)
            restored = (RolePipeline.from_states(steps, model.export_states())
                        if kind == "role" else MultimodalPipeline.from_state(
                            model.export_state(), recipe=recipe, source_schemas=schemas
                        ))
            np.testing.assert_array_equal(restored.predict(inputs), expected)
            pair = [model._handle_.value, restored._handle_.value] if kind == "role" else [
                model._handle.value, restored._handle.value
            ]
            owner = threading.get_ident()
            with lock:
                lifetimes = [(owner, handle) for handle in pair]
                owners.extend(lifetimes)
                tracked_lifetimes.update(lifetimes)
            ready_close.wait()
        finally:
            if restored is not None:
                restored.close()
                restored.close()
            model.close()
            model.close()
        with pytest.raises((N4MError, RuntimeError), match="not fitted|closed"):
            model.predict(inputs)

    with witness(monkeypatch, serialized=serialized) as probe:
        with monkeypatch.context() as scoped:
            scoped.setattr(lib, destroy_symbol, observed_destroy)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(worker) for _ in range(workers)]
                for future in futures:
                    future.result(timeout=120)
        stats = probe.snapshot()
    records = stats["records"]
    assert stats["total_calls"] == workers, "Hydration/replay unexpectedly called FIT"
    assert len({row["handle"] for row in records}) == workers
    if serialized:
        assert stats["peak_active"] == 1, "Serialized native negative control reported overlap"
    else:
        assert stats["peak_active"] >= 2, "No overlapping in-flight native FIT probes observed"
        assert len({row["native_thread"] for row in records}) >= 2
        for left in records:
            for right in records:
                if left is not right and max(left["entered_ns"], right["entered_ns"]) < min(
                    left["exited_ns"], right["exited_ns"]
                ):
                    assert left["context"] != right["context"]
                    assert left["handle"] != right["handle"]
    # Every retained model is live together at ready_close; addresses cannot be
    # reused before this assertion's lifetime boundary. Ignore unretained
    # Multimodal constructor/import placeholders: start observing each retained
    # allocation only after it is published, while every model remains alive.
    assert len({handle for _, handle in owners}) == workers * 2
    assert all(destroyed.count(owner) == 1 for owner in owners)
    assert input_hashes(blocks, y) == before
    assert all(not value.flags.writeable for value in [*blocks.values(), y])
    record_property("native_fit_probe_stats", json.dumps(stats))
    record_property("native_fit_probe_serialized_control", serialized)
    record_property("shared_input_hashes", json.dumps(before, sort_keys=True))
    record_property("process_peak_rss_native_units", peak_rss())


@pytest.mark.parametrize("kind", ["role", "multimodal"])
def test_failed_worker_closes_while_successful_sibling_replays(kind):
    blocks, y, recipe, schemas = shared_case()
    start = threading.Barrier(2, timeout=30)
    steps = ["models.regularized.ridge"]
    models = []
    lock = threading.Lock()

    def worker(fail):
        pipeline = (RolePipeline(steps) if kind == "role"
                    else MultimodalPipeline(recipe, schemas))
        with lock:
            models.append(pipeline)
        restored = None
        try:
            start.wait()
            if fail:
                if kind == "role":
                    with pytest.raises(N4MError, match="not used"):
                        pipeline.fit(blocks["nir"], y, groups=np.arange(len(y)))
                else:
                    invalid_y = y.copy()
                    invalid_y[0] = np.nan
                    with pytest.raises(RuntimeError, match="NaN|Inf"):
                        pipeline.fit(blocks, invalid_y)
                return "failed_as_expected"
            pipeline.fit(blocks["nir"] if kind == "role" else blocks, y)
            inputs = blocks["nir"] if kind == "role" else blocks
            expected = pipeline.predict(inputs)
            restored = (RolePipeline.from_states(steps, pipeline.export_states())
                        if kind == "role" else MultimodalPipeline.from_state(
                            pipeline.export_state(), recipe=recipe, source_schemas=schemas
                        ))
            np.testing.assert_array_equal(restored.predict(inputs), expected)
            assert np.isfinite(expected).all()
            return "successful_sibling"
        finally:
            if restored is not None:
                restored.close()
                restored.close()
            pipeline.close()
            pipeline.close()

    before = input_hashes(blocks, y)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, fail) for fail in (True, False)]
        results = [future.result(timeout=120) for future in futures]
    assert results == ["failed_as_expected", "successful_sibling"]
    for pipeline in models:
        assert pipeline._handle_ is None if kind == "role" else not pipeline._handle.value
    assert input_hashes(blocks, y) == before


def test_missing_mandatory_native_probe_is_not_skipped(monkeypatch):
    monkeypatch.delenv("N4M_NATIVE_FIT_WITNESS_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="Mandatory native FIT witness"):
        with witness(monkeypatch):
            pytest.fail("Missing probe must refuse entry")


def test_bounded_native_probe_overflow_refuses_partial_evidence(monkeypatch):
    x = np.arange(48, dtype=np.float64).reshape(16, 3)
    y = x[:, 0] - x[:, 1]
    pipeline = RolePipeline(["models.regularized.ridge"])
    try:
        with witness(monkeypatch, capacity=1) as probe:
            pipeline.fit(x, y)
            pipeline.fit(x, y)
            with pytest.raises(RuntimeError, match="bounded-store overflow"):
                probe.snapshot()
    finally:
        pipeline.close()
