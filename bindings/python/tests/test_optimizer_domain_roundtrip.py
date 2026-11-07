# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Public optimizer domain, checkpoint, and trajectory regressions."""

import ctypes as ct
import math
import random
import struct
import sys
from fractions import Fraction

import pytest
from n4m import Context
from n4m._errors import N4MError
from n4m._ffi import lib
from n4m.model_selection.optimizer import (
    ConstraintKind,
    Direction,
    Optimizer,
    Sampler,
    SearchSpace,
    TrialError,
    TrialStatus,
)

# Genuine pre-fix EMPTY checkpoints from the independently retained Opus evidence.
EMPTY_FLOAT = bytes.fromhex(
    "4e344d4f50540d0a0100000020000000e800000000000000c0000000000000002a4561ca5cd613b13c0000000000000001000000010000007801000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000eb9af1ffa822dc3e00000000000000000000000000000000000000000000000000000000e41aeb13812ec382"
)
EMPTY_TUPLE = bytes.fromhex(
    "4e344d4f50540d0a0100000020000000e800000000000000c00000000000000061667f6d1e6e72713c0000000000000001000000010000007806000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000200000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000fb3f8cb2ac6bdc3e000000000000000000000000000000000000000000000000000000008f9e37096cfc04b1"
)

MAX = sys.float_info.max
POW = 2.0**1023
BAD = [(-1.7e308, 1.7e308), (-MAX, MAX), (-POW, MAX / 2), (-POW, POW)]
GOOD = [
    (-MAX / 2, MAX / 2),
    (0.0, MAX),
    (-8e307, 8e307),
    (-POW, math.nextafter(MAX / 2, 0)),
]
SAMPLERS = [
    Sampler.RANDOM,
    Sampler.SOBOL,
    Sampler.LHS,
    Sampler.TERNARY,
    Sampler.GA,
    Sampler.PSO,
    Sampler.CMAES,
    Sampler.TPE,
    Sampler.GP_EI,
]
P = 2**52
TUPLES = [
    (0, 10),
    (-5, 5),
    (1000, 1002),
    (-1000, -998),
    (2**40, 2**40 + 2),
    (2**50, 2**50 + 2),
    (2**51, 2**51 + 2),
    (-(2**51) - 2, -(2**51)),
    (P, P),
    (P, P + 1),
    (P, P + 2),
    (P + 1000, P + 1003),
    (2**53 - 3, 2**53 - 1),
    (2**53 - 2, 2**53),
    (-(2**53), -(2**53) + 2),
    (-(2**53), -(2**53)),
    (-P - 2, -P),
    (-(2**53), 2**53),
    (-(2**53) + 1, 2**53 - 1),
]
LOG_DOMAINS = [
    (False, 0.1, math.nextafter(0.1, 1)),
    (False, 1e300, 1e300 * (1 + 1e-14)),
    (False, 1e-5, math.nextafter(math.nextafter(1e-5, 1), 1)),
    (True, 10**15, 10**15 + 1),
    (True, 10**15, 10**15 + 2),
    (True, 2**53 - 2, 2**53),
    (True, 10**15, 10**15 + 3),
    (False, 1.0, math.nextafter(1.0, 2.0)),
    (False, 5.0, 5.0),
    (False, 0.5, 2.5),
    (False, 1.0, 1000.0),
]


def bits(value):
    return struct.pack(">d", float(value))


def history(opt):
    return [
        (
            r.id,
            int(r.status),
            None if r.score is None else bits(r.score),
            tuple((k, bits(v)) for k, v in r.params.items()),
            r.ask_sequence,
            r.terminal_sequence,
            r.rung,
            tuple(
                (x.sequence, x.step, bits(x.score), x.should_prune)
                for x in r.intermediates
            ),
        )
        for r in opt.get_trials()
    ]


def stage_check(factory, sampler, seed, axis_names, validate, int_names=()):
    # Replay from the start for each stage so the reference remains a live owner.
    for count, running in [
        (0, False),
        (1, True),
        (1, False),
        (12, True),
        (12, False),
        (13, True),
    ]:
        space = factory()
        opt = Optimizer(space, sampler=sampler, seed=seed, direction=Direction.MINIMIZE)
        try:
            pending = None
            for j in range(count):
                pending = opt.ask()
                validate(pending)
                if not (running and j == count - 1):
                    opt.tell(pending.id, (j % 7) / 7)
            captured = history(opt)
            copy = Optimizer.load(opt.save())
            try:
                assert copy.configuration_matches(opt)
                assert history(copy) == captured
                if running:
                    opt.tell(pending.id, 0.5)
                    copy.tell(pending.id, 0.5)
                for j in range(6):
                    a, b = opt.ask(), copy.ask()
                    validate(a)
                    validate(b)
                    assert a.id == b.id
                    for name in axis_names:
                        assert bits(a.get_float(name)) == bits(b.get_float(name))
                    for name in int_names:
                        assert a.get_int(name) == b.get_int(name)
                    opt.tell(a.id, (j % 7) / 7)
                    copy.tell(b.id, (j % 7) / 7)
            finally:
                copy.close()
        finally:
            opt.close()
            space.close()


@pytest.mark.parametrize("low,high", BAD)
@pytest.mark.parametrize("tuple_axis", [False, True])
@pytest.mark.parametrize("aux", [False, True])
def test_overflow_domain_refused(low, high, tuple_axis, aux):
    space = SearchSpace()
    if tuple_axis:
        space.add_sorted_tuple("x", 2, low, high)
    else:
        space.add_float("x", low, high)
    if aux:
        space.add_float("aux", 0, 1)
    try:
        with pytest.raises(N4MError, match="INVALID_ARGUMENT"):
            Optimizer(space, seed=19)
    finally:
        space.close()


@pytest.mark.parametrize("checkpoint", [EMPTY_FLOAT, EMPTY_TUPLE])
def test_genuine_old_overflow_checkpoint_refused(checkpoint):
    with pytest.raises(N4MError, match="CORRUPT_BUFFER"):
        Optimizer.load(checkpoint)


@pytest.mark.parametrize("low,high", GOOD)
@pytest.mark.parametrize("tuple_axis", [False, True])
def test_finite_span_checkpoint_control(low, high, tuple_axis):
    space = SearchSpace()
    if tuple_axis:
        space.add_sorted_tuple("x", 2, low, high)
    else:
        space.add_float("x", low, high)
    opt = Optimizer(space, seed=19)
    try:
        copy = Optimizer.load(opt.save())
        try:
            assert copy.configuration_matches(opt)
            assert history(copy) == []
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("integer,low,high", LOG_DOMAINS)
@pytest.mark.parametrize("sampler", [Sampler.GP_EI, Sampler.TPE])
@pytest.mark.parametrize("seed", [19, 3])
def test_collapsed_log_staged(integer, low, high, sampler, seed):
    def factory():
        space = SearchSpace()
        if integer:
            space.add_int("x", low, high, log=True)
        else:
            space.add_float("x", low, high, log=True)
        return space.add_float("aux", 0, 1)

    def validate(trial):
        x = trial.get_int("x") if integer else trial.get_float("x")
        assert math.isfinite(x) and low <= x <= high
        assert 0 <= trial.get_float("aux") <= 1

    # get_float is defined only for float axes: integer axis comparison is exact.
    stage_check(
        factory,
        sampler,
        seed,
        ["aux"] if integer else ["x", "aux"],
        validate,
        ["x"] if integer else [],
    )


@pytest.mark.parametrize("low,high", TUPLES)
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("length", [1, 3])
def test_integer_tuple_staged(low, high, sampler, length):
    def factory():
        return (
            SearchSpace()
            .add_sorted_tuple("x", length, low, high, element_is_int=True)
            .add_float("aux", 0, 1)
        )

    names = [f"x#{j}" for j in range(length)] + ["aux"]

    def validate(trial):
        values = [trial.get_float(f"x#{j}") for j in range(length)]
        assert all(
            math.isfinite(v) and v == math.trunc(v) and low <= v <= high for v in values
        )
        assert values == sorted(values)

    stage_check(factory, sampler, 19, names, validate)


@pytest.mark.parametrize(
    "low,high",
    [(low, high) for low, high in TUPLES if high < 2**53 and high - low < 2**53],
)
def test_random_tuple_oracle_and_uniformity(low, high):
    eligible = [(lo, hi) for lo, hi in TUPLES if hi < 2**53 and hi - lo < 2**53]
    assert len(eligible) == 16
    assert set(TUPLES) - set(eligible) == {
        (2**53 - 2, 2**53),
        (-(2**53), 2**53),
        (-(2**53) + 1, 2**53 - 1),
    }
    n = 6000

    def scalar():
        space = SearchSpace().add_int("x", low, high)
        opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
        try:
            values = []
            for _ in range(n):
                trial = opt.ask()
                values.append(trial.get_int("x"))
                opt.tell(trial.id, 0.0)
            return values
        finally:
            opt.close()
            space.close()

    def tuples(length, count):
        space = SearchSpace().add_sorted_tuple(
            "x", length, low, high, element_is_int=True
        )
        opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
        try:
            values = []
            for _ in range(count):
                trial = opt.ask()
                values.append([trial.get_float(f"x#{j}") for j in range(length)])
                opt.tell(trial.id, 0.0)
            return values
        finally:
            opt.close()
            space.close()

    original = scalar()
    singles = [v[0] for v in tuples(1, n)]
    triples = tuples(3, n // 3)
    assert singles == original
    assert triples == [sorted(original[i : i + 3]) for i in range(0, n, 3)]
    if (low, high) == (2**52, 2**52 + 1):
        assert abs(sum(v == high for v in singles) / n - 0.5) <= 0.03


@pytest.mark.parametrize("sampler", [Sampler.RANDOM, Sampler.TPE, Sampler.GP_EI])
def test_log_enqueue_contract(sampler):
    low, high = 10**15, 10**15 + 2
    space = SearchSpace().add_int("x", low, high, log=True).add_float("aux", 0, 1)
    opt = Optimizer(space, sampler=sampler, seed=19)
    try:
        if sampler == Sampler.GP_EI:
            with pytest.raises(N4MError, match="UNSUPPORTED"):
                opt.enqueue({"x": low, "aux": 0.25})
            return
        for x in (low, high, low + 1):
            opt.enqueue({"x": x, "aux": 0.25})
        for x in (low, high, low + 1):
            trial = opt.ask()
            assert trial.get_int("x") == x
            opt.tell(trial.id, 0.5)
        copy = Optimizer.load(opt.save())
        try:
            assert copy.configuration_matches(opt)
            assert history(copy) == history(opt)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


def test_gp_proposal_checkpoint_guard():
    space = SearchSpace().add_float("x", 0.0, 1.0)
    opt = Optimizer(space, sampler=Sampler.GP_EI, seed=19)
    try:
        trial = opt.ask()
        raw = opt.save()
        pattern = struct.pack("<d", trial.get_float("x"))
        positions = [i for i in range(len(raw) - 8) if raw[i : i + 8] == pattern]
        assert len(positions) == 2
        for value in (math.nan, -0.1, 1.5):
            damaged = bytearray(raw)
            damaged[positions[1] : positions[1] + 8] = struct.pack("<d", value)
            checksum = 14695981039346656037
            for byte in damaged[:-8]:
                checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
            damaged[-8:] = checksum.to_bytes(8, "little")
            with pytest.raises(N4MError, match="CORRUPT_BUFFER"):
                Optimizer.load(damaged)
    finally:
        opt.close()
        space.close()


def test_gp_collapsed_domain_stored_unit_bits():
    space = SearchSpace().add_float("x", 0.1, math.nextafter(0.1, 1.0), log=True)
    opt = Optimizer(space, sampler=Sampler.GP_EI, seed=19)
    try:
        for count in range(1, 4):
            trial = opt.ask()
            opt.tell(trial.id, 0.5)
            checkpoint = opt.save()
            marker = struct.pack("<Q", 0x3FEFFFFFFFFFFFFF)
            hits = sum(
                checkpoint[i : i + 8] == marker for i in range(len(checkpoint) - 7)
            )
            assert hits == count
            copy = Optimizer.load(checkpoint)
            copy.close()
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("low,high", GOOD)
@pytest.mark.parametrize("tuple_axis", [False, True])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_finite_span_staged(low, high, tuple_axis, sampler):
    def factory():
        space = SearchSpace()
        if tuple_axis:
            space.add_sorted_tuple("x", 2, low, high)
        else:
            space.add_float("x", low, high)
        return space.add_float("aux", 0, 1)

    names = (["x#0", "x#1"] if tuple_axis else ["x"]) + ["aux"]

    def validate(trial):
        values = [trial.get_float(n) for n in names[:-1]]
        assert all(math.isfinite(v) and low <= v <= high for v in values)
        assert values == sorted(values)

    stage_check(factory, sampler, 19, names, validate)


def test_float_span_oracle():
    rng = random.Random(19)
    checked = 0
    while checked < 1000:
        low = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        high = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        if not (math.isfinite(low) and math.isfinite(high)):
            continue
        low, high = sorted((low, high))
        reject = high / 2 - low / 2 > MAX / 2
        for tuple_axis in (False, True):
            space = SearchSpace()
            if tuple_axis:
                space.add_sorted_tuple("x", 2, low, high)
            else:
                space.add_float("x", low, high)
            try:
                if reject:
                    with pytest.raises(N4MError, match="INVALID_ARGUMENT"):
                        Optimizer(space, seed=19)
                else:
                    opt = Optimizer(space, seed=19)
                    opt.close()
            finally:
                space.close()
        checked += 1


@pytest.mark.parametrize("sampler", [Sampler.RANDOM, Sampler.TPE, Sampler.GP_EI])
def test_log_float_enqueue_contract(sampler):
    low, high = 1e300, 1e300 * (1 + 1e-14)
    middle = low + (high - low) / 2
    assert low < middle < high
    space = SearchSpace().add_float("x", low, high, log=True).add_float("aux", 0, 1)
    opt = Optimizer(space, sampler=sampler, seed=19)
    try:
        if sampler == Sampler.GP_EI:
            with pytest.raises(N4MError, match="UNSUPPORTED"):
                opt.enqueue({"x": low, "aux": 0.25})
        else:
            for x in (low, high, middle):
                opt.enqueue({"x": x, "aux": 0.25})
            for x in (low, high, middle):
                trial = opt.ask()
                assert bits(trial.get_float("x")) == bits(x)
                opt.tell(trial.id, 0.5)
    finally:
        opt.close()
        space.close()


# T4e: the independent reference uses the public RANDOM raw-unit stream and
# Python integer arithmetic, never the production grid helpers.
H = 2**53
SCALAR_INTEGER_GRIDS = [
    (H - 2, H, 1),
    (H - 3, H, 2),
    (H - 5, H, 3),
    (2**52, 2**52 + 7, 8),
    (2**49, 2**49 + 3, 4),
    (2**48, 2**48 + 3, 4),
    (-H, H, 1),
    (-H, H, 3),
    (-H, H, 7),
    (-1, H, 1),
    (-3, H, 1),
    (0, H, 1),
    (0, H - 2, 1),
    (-H, H, 5),
    (-H, H, 9),
    (-H, H, 11),
]
SCALAR_NEAR_ULP = [
    (1 - 2**-52, 1.0, 2**-53, 3),
    (0.5 - 2**-53, 0.5, 2**-54, 3),
    (float(2**60 - 256), float(2**60), 128.0, 3),
    (1 - 3 * 2**-53, 1.0, 2**-53, 4),
    (float(2**52), float(2**52 + 4), 1.5, 3),
]


def scalar_space(integer, low, high, step):
    space = SearchSpace()
    if integer:
        space.add_int("x", low, high, step)
    else:
        space.add_float("x", float(low), float(high), float(step))
    return space.add_float("aux", 0, 1)


@pytest.mark.parametrize("integer", [True, False])
@pytest.mark.parametrize("low,high,step", SCALAR_INTEGER_GRIDS)
def test_scalar_grid_random_bigint_oracle(integer, low, high, step):
    space = scalar_space(integer, low, high, step)
    raw_space = SearchSpace().add_float("x", 0, 1).add_float("aux", 0, 1)
    grid = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    raw = Optimizer(raw_space, sampler=Sampler.RANDOM, seed=19)
    counts = {}
    try:
        count = (high - low) // step + 1
        for _ in range(6000):
            a, b = grid.ask(), raw.ask()
            product = Fraction(b.get_float("x")) * count
            index = math.floor(float(product)) if count - 1 < H else product.__floor__()
            expected = low + step * index
            actual = a.get_int("x") if integer else a.get_float("x")
            assert low <= actual <= high
            assert actual == expected
            assert bits(a.get_float("aux")) == bits(b.get_float("aux"))
            counts[actual] = counts.get(actual, 0) + 1
            grid.tell(a.id, 0.0)
            raw.tell(b.id, 0.0)
    finally:
        grid.close()
        raw.close()
        space.close()
        raw_space.close()


@pytest.mark.parametrize("low,high,step,cells", SCALAR_NEAR_ULP)
def test_scalar_grid_near_ulp_distribution(low, high, step, cells):
    space = scalar_space(False, low, high, step)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    raw_space = SearchSpace().add_float("x", 0, 1).add_float("aux", 0, 1)
    raw = Optimizer(raw_space, sampler=Sampler.RANDOM, seed=19)
    counts = {}
    try:
        for _ in range(6000):
            trial, reference = opt.ask(), raw.ask()
            assert bits(trial.get_float("aux")) == bits(reference.get_float("aux"))
            value = trial.get_float("x")
            assert low <= value <= high
            counts[bits(value)] = counts.get(bits(value), 0) + 1
            opt.tell(trial.id, 0.0)
            raw.tell(reference.id, 0.0)
        assert len(counts) == cells
        assert all(abs(count / 6000 - 1 / cells) <= 0.03 for count in counts.values())
    finally:
        opt.close()
        raw.close()
        space.close()
        raw_space.close()


@pytest.mark.parametrize("integer", [True, False])
@pytest.mark.parametrize("low,high,step", [(H - 3, H, 2), (-H, H, 3)])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_scalar_grid_sampler_staged(integer, low, high, step, sampler):
    def factory():
        return scalar_space(integer, low, high, step)

    def validate(trial):
        value = trial.get_int("x") if integer else trial.get_float("x")
        assert low <= value <= high and int(value) == value
        assert (int(value) - low) % step == 0
        assert 0 <= trial.get_float("aux") <= 1

    stage_check(
        factory,
        sampler,
        19,
        ["aux"] if integer else ["x", "aux"],
        validate,
        ["x"] if integer else [],
    )


def test_scalar_grid_enqueue_exact_integer_float():
    for low, high, step, accepted, rejected in [
        (-H, H, 3, [1, 7, 13, H - 1], []),
        (2**52, 2**52 + 16, 8, [2**52 + 8], [2**52 + 1, 2**52 + 7, 2**52 + 9]),
        (H - 3, H, 2, [], [H]),
    ]:
        space = SearchSpace().add_float("x", low, high, step)
        opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
        try:
            for value in rejected:
                with pytest.raises(N4MError, match="INVALID_ARGUMENT"):
                    opt.enqueue({"x": value})
            for value in accepted:
                opt.enqueue({"x": value})
                trial = opt.ask()
                assert bits(trial.get_float("x")) == bits(value)
                opt.tell(trial.id, 0.0)
        finally:
            opt.close()
            space.close()


# D0 public-binding control bits, seed 19, pinned from retained f384.
SCALAR_CONTROL_VECTORS = [
    (
        -9007199254740992,
        -9007199254740990,
        1,
        [
            "c33ffffffffffffe",
            "c340000000000000",
            "c33fffffffffffff",
            "c340000000000000",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
            "c33fffffffffffff",
            "c33fffffffffffff",
            "c340000000000000",
            "c33fffffffffffff",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
            "c33ffffffffffffe",
        ],
    ),
    (
        1000,
        1002,
        1,
        [
            "408f500000000000",
            "408f400000000000",
            "408f480000000000",
            "408f400000000000",
            "408f500000000000",
            "408f500000000000",
            "408f480000000000",
            "408f480000000000",
            "408f400000000000",
            "408f480000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
        ],
    ),
    (
        9007199254740989,
        9007199254740991,
        1,
        [
            "433fffffffffffff",
            "433ffffffffffffd",
            "433ffffffffffffe",
            "433ffffffffffffd",
            "433fffffffffffff",
            "433fffffffffffff",
            "433ffffffffffffe",
            "433ffffffffffffe",
            "433ffffffffffffd",
            "433ffffffffffffe",
            "433fffffffffffff",
            "433fffffffffffff",
            "433fffffffffffff",
            "433fffffffffffff",
            "433fffffffffffff",
            "433fffffffffffff",
        ],
    ),
    (
        9007199254740988,
        9007199254740992,
        2,
        [
            "4340000000000000",
            "433ffffffffffffc",
            "433ffffffffffffe",
            "433ffffffffffffc",
            "4340000000000000",
            "4340000000000000",
            "433ffffffffffffe",
            "433ffffffffffffe",
            "433ffffffffffffc",
            "433ffffffffffffe",
            "4340000000000000",
            "4340000000000000",
            "4340000000000000",
            "4340000000000000",
            "4340000000000000",
            "4340000000000000",
        ],
    ),
    (
        140737488355328,
        140737488355331,
        4,
        [
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
            "42e0000000000000",
        ],
    ),
    (
        1000,
        1003,
        2,
        [
            "408f500000000000",
            "408f400000000000",
            "408f400000000000",
            "408f400000000000",
            "408f500000000000",
            "408f500000000000",
            "408f400000000000",
            "408f500000000000",
            "408f400000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
            "408f500000000000",
        ],
    ),
    (
        0.0,
        1.0,
        0.1,
        [
            "3fe999999999999a",
            "3fb999999999999a",
            "3fd3333333333334",
            "0000000000000000",
            "3fe999999999999a",
            "3ff0000000000000",
            "3fe0000000000000",
            "3fe3333333333334",
            "3fc999999999999a",
            "3fe0000000000000",
            "3feccccccccccccd",
            "3ff0000000000000",
            "3fe999999999999a",
            "3feccccccccccccd",
            "3fe999999999999a",
            "3fe999999999999a",
        ],
    ),
    (
        0.0,
        0.7,
        0.1,
        [
            "3fe0000000000000",
            "0000000000000000",
            "3fc999999999999a",
            "0000000000000000",
            "3fe3333333333334",
            "3fe6666666666666",
            "3fd3333333333334",
            "3fd999999999999a",
            "3fb999999999999a",
            "3fd999999999999a",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fe3333333333334",
            "3fe3333333333334",
            "3fe3333333333334",
            "3fe0000000000000",
        ],
    ),
    (
        0.1,
        0.7,
        0.2,
        [
            "3fe0000000000000",
            "3fb999999999999a",
            "3fd3333333333334",
            "3fb999999999999a",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fd3333333333334",
            "3fe0000000000000",
            "3fb999999999999a",
            "3fe0000000000000",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fe6666666666666",
            "3fe0000000000000",
        ],
    ),
    (
        0.0,
        0.3,
        0.1,
        [
            "3fc999999999999a",
            "0000000000000000",
            "3fb999999999999a",
            "0000000000000000",
            "3fd3333333333333",
            "3fd3333333333333",
            "3fb999999999999a",
            "3fc999999999999a",
            "0000000000000000",
            "3fc999999999999a",
            "3fd3333333333333",
            "3fd3333333333333",
            "3fd3333333333333",
            "3fd3333333333333",
            "3fd3333333333333",
            "3fc999999999999a",
        ],
    ),
    (
        -1.0,
        1.0,
        0.25,
        [
            "3fe0000000000000",
            "bfe8000000000000",
            "bfd0000000000000",
            "bff0000000000000",
            "3fe8000000000000",
            "3ff0000000000000",
            "0000000000000000",
            "3fd0000000000000",
            "bfe0000000000000",
            "0000000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "3fe8000000000000",
            "3fe8000000000000",
            "3fe8000000000000",
            "3fe0000000000000",
        ],
    ),
    (
        5e-324,
        2.5e-323,
        5e-324,
        [
            "0000000000000004",
            "0000000000000001",
            "0000000000000002",
            "0000000000000001",
            "0000000000000004",
            "0000000000000005",
            "0000000000000003",
            "0000000000000003",
            "0000000000000002",
            "0000000000000003",
            "0000000000000005",
            "0000000000000005",
            "0000000000000005",
            "0000000000000005",
            "0000000000000004",
            "0000000000000004",
        ],
    ),
    (
        0.9999999999999998,
        1.0,
        2.220446049250313e-16,
        [
            "3ff0000000000000",
            "3feffffffffffffe",
            "3feffffffffffffe",
            "3feffffffffffffe",
            "3ff0000000000000",
            "3ff0000000000000",
            "3feffffffffffffe",
            "3ff0000000000000",
            "3feffffffffffffe",
            "3ff0000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
            "3ff0000000000000",
        ],
    ),
    (
        1.152921504606847e18,
        1.152921504606848e18,
        200.0,
        [
            "43b0000000000003",
            "43b0000000000000",
            "43b0000000000002",
            "43b0000000000000",
            "43b0000000000003",
            "43b0000000000004",
            "43b0000000000002",
            "43b0000000000002",
            "43b0000000000001",
            "43b0000000000002",
            "43b0000000000004",
            "43b0000000000004",
            "43b0000000000003",
            "43b0000000000004",
            "43b0000000000003",
            "43b0000000000003",
        ],
    ),
]


@pytest.mark.parametrize("low,high,step,expected", SCALAR_CONTROL_VECTORS)
def test_scalar_grid_unchanged_control_bits(low, high, step, expected):
    space = SearchSpace().add_float("x", low, high, step)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        for value in expected:
            trial = opt.ask()
            assert bits(trial.get_float("x")).hex() == value
            opt.tell(trial.id, 0.0)
    finally:
        opt.close()
        space.close()


def test_scalar_grid_integer_enqueue_high_refused():
    space = SearchSpace().add_int("x", H - 3, H, 2)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        with pytest.raises(N4MError, match="INVALID_ARGUMENT"):
            opt.enqueue({"x": H})
    finally:
        opt.close()
        space.close()


def test_scalar_grid_decimal_enqueue_snapping():
    space = SearchSpace().add_float("x", 0, 1, 0.1)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": 0.1 + 0.2})
        trial = opt.ask()
        assert bits(trial.get_float("x")) == bits(0.1 * 3)
        opt.tell(trial.id, 0.0)
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("integer", [True, False])
def test_scalar_grid_exact_distribution(integer):
    space = SearchSpace()
    if integer:
        space.add_int("x", H - 2, H, 1)
    else:
        space.add_float("x", H - 2, H, 1)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    counts = [0, 0, 0]
    try:
        for _ in range(6000):
            trial = opt.ask()
            x = trial.get_int("x") if integer else int(trial.get_float("x"))
            assert H - 2 <= x <= H
            counts[x - (H - 2)] += 1
            opt.tell(trial.id, 0.0)
        assert counts == [2015, 2005, 1980]
    finally:
        opt.close()
        space.close()


# T4f: compare nearest-representable grid values, not exact-rational membership.
T4F_C = 2.0**-53
T4F_R1 = [(-1.0, 1.0 - 2**-52, 3 * T4F_C), (-0.5, 0.5 - 2**-53, 1.5 * T4F_C)]
T4F_R2 = [
    (1.3094115832905704, 9.007642868376131e21, 3.020767460720702),
    (1.3552258541085935, 3.1816648682723785e21, 9.932851196532727),
]
T4F_R4 = [
    (-1.0, 1.0, 3 * T4F_C),
    (-(2.0**43), 2.0**43, 3 / 1024),
    (-(2.0**54), 2.0**54, 6.0),
]
T4F_R3 = T4F_R4 + [(-1.0, 1.0, 1.25 * T4F_C), (-1.0, 1.0, 1.5 * T4F_C)]


@pytest.mark.parametrize(
    "sampler", [Sampler.CMAES, Sampler.PSO, Sampler.GA, Sampler.TPE]
)
@pytest.mark.parametrize(
    "low,high,step,count", [(*d, 400) for d in T4F_R1] + [(*d, 240) for d in T4F_R2]
)
def test_t4f_r1_r2_bounded_checkpoint(sampler, low, high, step, count):
    space = SearchSpace().add_float("x", low, high, step).add_float("aux", 0, 1)
    opt = Optimizer(space, sampler=sampler, seed=19, direction=Direction.MINIMIZE)
    try:
        outside = []
        for i in range(count):
            t = opt.ask()
            x = t.get_float("x")
            if not low <= x <= high:
                outside.append((i, x))
            opt.tell(t.id, -x)
        assert not outside, outside[:3]
        copy = Optimizer.load(opt.save())
        try:
            assert copy.configuration_matches(opt) and history(copy) == history(opt)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("low,high,step", T4F_R3)
def test_t4f_r3_own_enqueue(low, high, step):
    s = SearchSpace().add_float("x", low, high, step)
    v = SearchSpace().add_float("x", low, high, step)
    original = Optimizer(s, sampler=Sampler.RANDOM, seed=19)
    verify = Optimizer(v, sampler=Sampler.RANDOM, seed=3)
    try:
        for _ in range(300):
            t = original.ask()
            x = t.get_float("x")
            verify.enqueue({"x": x})
            q = verify.ask()
            assert bits(q.get_float("x")) == bits(x)
            original.tell(t.id, 0)
            verify.tell(q.id, 0)
    finally:
        original.close()
        verify.close()
        s.close()
        v.close()


@pytest.mark.parametrize("low,high,step", T4F_R4)
def test_t4f_r4_fraction_oracle(low, high, step):
    q = (Fraction(high) - Fraction(low)) // Fraction(step)
    n = q + 1
    s = SearchSpace().add_float("x", low, high, step)
    u = SearchSpace().add_float("x", 0, 1)
    grid = Optimizer(s, sampler=Sampler.RANDOM, seed=19)
    raw = Optimizer(u, sampler=Sampler.RANDOM, seed=19)
    try:
        for _ in range(6000):
            t, z = grid.ask(), raw.ask()
            index = math.floor(float(Fraction(z.get_float("x")) * n))
            expected = float(Fraction(low) + index * Fraction(step))
            x = t.get_float("x")
            assert bits(x) == bits(expected)
            grid.tell(t.id, 0)
            raw.tell(z.id, 0)
    finally:
        grid.close()
        raw.close()
        s.close()
        u.close()


@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("low,high,step", T4F_R1 + T4F_R3)
def test_t4f_staged_checkpoint(sampler, low, high, step):
    def factory():
        return SearchSpace().add_float("x", low, high, step).add_float("aux", 0, 1)

    def validate(t):
        x = t.get_float("x")
        assert math.isfinite(x) and low <= x <= high

    stage_check(factory, sampler, 19, ["x", "aux"], validate)


@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("low,high,step", T4F_R1 + T4F_R3)
def test_t4f_queued_checkpoint(sampler, low, high, step):
    space = SearchSpace().add_float("x", low, high, step).add_float("aux", 0, 1)
    opt = Optimizer(space, sampler=sampler, seed=19)
    try:
        if sampler in (Sampler.GA, Sampler.PSO, Sampler.CMAES, Sampler.GP_EI):
            with pytest.raises(N4MError, match="UNSUPPORTED"):
                opt.enqueue({"x": low, "aux": 0.25})
            return
        opt.enqueue({"x": low, "aux": 0.25})
        copy = Optimizer.load(opt.save())
        try:
            assert copy.configuration_matches(opt) and history(copy) == history(opt)
            for j in range(6):
                a, b = opt.ask(), copy.ask()
                assert a.id == b.id
                for name in ("x", "aux"):
                    assert bits(a.get_float(name)) == bits(b.get_float(name))
                if j == 0:
                    assert bits(a.get_float("x")) == bits(low)
                opt.tell(a.id, 0.5)
                copy.tell(b.id, 0.5)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


# Loader scalar-grid validation: locate a named trial record, never an
# arbitrary matching f64 in the search-space or sampler sections.
def grid_checkpoint_patch(
    blob, name, old, new, rank=0, active=True, expected_matches=None
):
    encoded = name.encode()
    marker = (
        struct.pack("<I", len(encoded))
        + encoded
        + struct.pack("<d", float(old))
        + struct.pack("<iI", -1, 0)
        + bytes([int(active)])
    )
    hits = [
        i
        for i in range(len(blob) - len(marker) + 1)
        if blob[i : i + len(marker)] == marker
    ]
    if expected_matches is not None:
        assert len(hits) == expected_matches
    assert 0 <= rank < len(hits)
    offset = hits[rank] + 4 + len(encoded)
    assert blob[offset : offset + 8] == struct.pack("<d", float(old))
    out = bytearray(blob)
    out[offset : offset + 8] = struct.pack("<d", float(new))
    assert out[offset : offset + 8] != blob[offset : offset + 8]
    fnv = 14695981039346656037
    for byte in out[:-8]:
        fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64) - 1)
    out[-8:] = fnv.to_bytes(8, "little")
    assert all(
        blob[i] == out[i]
        for i in range(len(blob))
        if not (offset <= i < offset + 8 or i >= len(blob) - 8)
    )
    return bytes(out)


def grid_assert_corrupt_null(blob):
    ctx = Context()
    handle = ct.c_void_p()
    data = (ct.c_uint8 * len(blob)).from_buffer_copy(blob)
    try:
        status = lib.n4m_optimizer_load(ctx._handle, data, len(blob), ct.byref(handle))
        try:
            assert status == 14 and not handle.value, (
                f"status={status}, handle_nonnull={bool(handle.value)}"
            )
        finally:
            if handle.value:
                lib.n4m_optimizer_destroy(handle)
    finally:
        ctx.close()


def grid_set_status(opt, trial, status):
    if status == TrialStatus.COMPLETED:
        opt.tell(trial.id, 0.25)
    elif status in (TrialStatus.FAILED, TrialStatus.CANCELLED):
        opt.tell_result(trial.id, status, error=TrialError("GRID_TEST", "fixture"))
    elif status == TrialStatus.PRUNED:
        opt.tell_result(trial.id, status)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize(
    "status",
    [
        TrialStatus.RUNNING,
        TrialStatus.COMPLETED,
        TrialStatus.FAILED,
        TrialStatus.CANCELLED,
        TrialStatus.PRUNED,
    ],
)
def test_loader_grid_history_status_and_rank(rank, status):
    space = SearchSpace().add_float("x", 0, 4, 2)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        for j in range(3):
            opt.enqueue({"x": 2})
            trial = opt.ask()
            assert trial.get_float("x") == 2
            grid_set_status(opt, trial, status if j == rank else TrialStatus.COMPLETED)
        before = opt.get_trials()
        assert before[rank].id == rank and before[rank].status == status
        damaged = grid_checkpoint_patch(opt.save(), "x", 2, 3, rank, expected_matches=3)
        grid_assert_corrupt_null(damaged)
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize(
    "integer,low,high,step,old,new",
    [
        (False, 0, 4, 2, 2, math.nextafter(2, math.inf)),
        (True, 0, 10, 3, 6, 7),
        (True, 0, 10, 3, 9, 10),
        (True, 2**53 - 3, 2**53, 2, 2**53 - 1, 2**53),
        (False, 0, 1, 0.1, 0.30000000000000004, 0.35),
        (False, 0, 1, 0.1, 0.30000000000000004, 0.3),
        (
            False,
            0,
            1,
            0.1,
            0.30000000000000004,
            math.nextafter(0.30000000000000004, math.inf),
        ),
        (False, -1, 1, 3 * 2**-53, -1, -1 + 2**-53),
        (False, 2**52, 2**52 + 16, 8, 2**52 + 8, 2**52 + 1),
    ],
)
def test_loader_grid_negative_domains(integer, low, high, step, old, new):
    space = SearchSpace()
    if integer:
        space.add_int("x", low, high, step)
    else:
        space.add_float("x", low, high, step)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": old})
        t = opt.ask()
        observed = t.get_int("x") if integer else t.get_float("x")
        assert bits(observed) == bits(old)
        opt.tell(t.id, 0.25)
        grid_assert_corrupt_null(
            grid_checkpoint_patch(opt.save(), "x", old, new, expected_matches=1)
        )
    finally:
        opt.close()
        space.close()


def test_loader_grid_inactive_field_rejected():
    space = (
        SearchSpace()
        .add_categorical("mode", ["off", "on"])
        .add_float("x", 1, 5, 2)
        .add_constraint(ConstraintKind.CONDITION_IN, ["x", "mode"], [None, "on"])
    )
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"mode": 0, "x": 3})
        t = opt.ask()
        assert not t.is_active("x") and t.get_float("x") == 3
        opt.tell(t.id, 0.25)
        grid_assert_corrupt_null(
            grid_checkpoint_patch(
                opt.save(), "x", 3, 4, active=False, expected_matches=1
            )
        )
    finally:
        opt.close()
        space.close()


def grid_positive_continuation(original, blob, expected_x):
    restored = Optimizer.load(blob)
    try:
        assert restored.configuration_matches(original)
        record = restored.get_trials()[0]
        assert bits(record.param_details["x"].value) == bits(expected_x)
        for j in range(6):
            a, b = original.ask(), restored.ask()
            assert a.id == b.id and bits(a.get_float("x")) == bits(b.get_float("x"))
            original.tell(a.id, j / 8)
            restored.tell(b.id, j / 8)
    finally:
        restored.close()


@pytest.mark.parametrize("old,new", [(2.0, 4.0), (0.0, -0.0)])
def test_loader_grid_valid_cell_and_negative_zero(old, new):
    space = SearchSpace().add_float("x", 0, 4, 2)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": old})
        t = opt.ask()
        assert bits(t.get_float("x")) == bits(old)
        opt.tell(t.id, 0.25)
        blob = grid_checkpoint_patch(opt.save(), "x", old, new, expected_matches=1)
        grid_positive_continuation(opt, blob, new)
    finally:
        opt.close()
        space.close()


def test_loader_grid_decimal_queue_replay():
    space = SearchSpace().add_float("x", 0, 1, 0.1)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": 0.3})
        raw = opt.save()
        old = struct.pack("<I", 1) + b"x" + struct.pack("<d", 0.30000000000000004)
        hits = [
            i for i in range(len(raw) - len(old) + 1) if raw[i : i + len(old)] == old
        ]
        assert len(hits) == 1

        def patch_queue(value):
            data = bytearray(raw)
            offset = hits[0] + 5
            data[offset : offset + 8] = struct.pack("<d", value)
            h = 14695981039346656037
            for byte in data[:-8]:
                h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
            data[-8:] = h.to_bytes(8, "little")
            assert all(
                raw[i] == data[i]
                for i in range(len(raw))
                if not (offset <= i < offset + 8 or i >= len(raw) - 8)
            )
            return bytes(data)

        restored = Optimizer.load(patch_queue(0.3))
        try:
            assert restored.configuration_matches(opt)
            a = restored.ask()
            assert bits(a.get_float("x")) == bits(0.30000000000000004)
            restored.tell(a.id, 0)
        finally:
            restored.close()
        grid_assert_corrupt_null(patch_queue(0.35))
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize(
    "kind,low,high,old,new",
    [
        ("log_int", 1, 1000, 10, 11),
        ("log_float", 1, 1000, 10, 11),
        ("continuous", 0, 4, 2, 3),
    ],
)
def test_loader_grid_unstepped_scalar_controls(kind, low, high, old, new):
    space = SearchSpace()
    if kind == "log_int":
        space.add_int("x", low, high, log=True)
    else:
        space.add_float("x", low, high, log=kind == "log_float")
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": old})
        t = opt.ask()
        opt.tell(t.id, 0.25)
        blob = grid_checkpoint_patch(opt.save(), "x", old, new, expected_matches=1)
        copy = Optimizer.load(blob)
        try:
            assert copy.configuration_matches(opt)
            assert bits(copy.get_trials()[0].param_details["x"].value) == bits(new)
            for j in range(6):
                a, b = opt.ask(), copy.ask()
                assert a.id == b.id
                if kind == "log_int":
                    assert a.get_int("x") == b.get_int("x")
                else:
                    assert bits(a.get_float("x")) == bits(b.get_float("x"))
                opt.tell(a.id, j / 8)
                copy.tell(b.id, j / 8)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


def test_loader_grid_sorted_tuple_control():
    space = SearchSpace().add_sorted_tuple("x", 2, 0, 4)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        t = opt.ask()
        old = t.get_float("x#0")
        other = t.get_float("x#1")
        assert 0 < old <= other
        opt.tell(t.id, 0.25)
        blob = grid_checkpoint_patch(opt.save(), "x#0", old, 0, expected_matches=1)
        copy = Optimizer.load(blob)
        try:
            assert copy.configuration_matches(opt) and bits(
                copy.get_trials()[0].param_details["x#0"].value
            ) == bits(0.0)
            for j in range(6):
                a, b = opt.ask(), copy.ask()
                assert a.id == b.id
                for name in ("x#0", "x#1"):
                    assert bits(a.get_float(name)) == bits(b.get_float(name))
                opt.tell(a.id, j / 8)
                copy.tell(b.id, j / 8)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


def test_loader_grid_categorical_signed_zero_control():
    space = SearchSpace().add_categorical("x", [0.0, 1.0])
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": 0})
        t = opt.ask()
        assert t.get_category("x")[0] == 0
        opt.tell(t.id, 0.25)
        raw = opt.save()
        marker = (
            struct.pack("<I", 1) + b"x" + struct.pack("<d", 0.0) + struct.pack("<i", 0)
        )
        hits = [
            i
            for i in range(len(raw) - len(marker) + 1)
            if raw[i : i + len(marker)] == marker
        ]
        assert len(hits) == 1
        offset = hits[0] + 5
        data = bytearray(raw)
        data[offset : offset + 8] = struct.pack("<d", -0.0)
        h = 14695981039346656037
        for byte in data[:-8]:
            h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
        data[-8:] = h.to_bytes(8, "little")
        assert all(
            raw[i] == data[i]
            for i in range(len(raw))
            if not (offset <= i < offset + 8 or i >= len(raw) - 8)
        )
        copy = Optimizer.load(bytes(data))
        try:
            assert copy.configuration_matches(opt) and bits(
                copy.get_trials()[0].param_details["x"].value
            ) == bits(-0.0)
            for j in range(6):
                a, b = opt.ask(), copy.ask()
                assert a.id == b.id and a.get_category("x") == b.get_category("x")
                opt.tell(a.id, j / 8)
                copy.tell(b.id, j / 8)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()


def test_loader_grid_enqueued_fraction_is_canonical():
    space = SearchSpace().add_float("x", 0, 1, 0.1)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        opt.enqueue({"x": 0.3})
        t = opt.ask()
        assert bits(t.get_float("x")) == bits(0.30000000000000004)
        opt.tell(t.id, 0.25)
        grid_positive_continuation(opt, opt.save(), 0.30000000000000004)
    finally:
        opt.close()
        space.close()


@pytest.mark.parametrize("low,high,step,_expected", SCALAR_CONTROL_VECTORS)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_loader_grid_control_checkpoint_stages(low, high, step, _expected, sampler):
    def factory():
        return SearchSpace().add_float("x", low, high, step).add_float("aux", 0, 1)

    def validate(t):
        assert low <= t.get_float("x") <= high

    stage_check(factory, sampler, 19, ["x", "aux"], validate)


# Genuine retained checkpoints; hashes and provenance are recorded in the receipt.
LOADER_RETAINED = {
    "valid.n4mopt": bytes.fromhex(
        "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000040ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90"
    ),
    "current-generated.n4mopt": bytes.fromhex(
        "4e344d4f50540d0a01000000200000007002000000000000480200000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a0000001300000000000000000000000000000000000000000000000000000067f029fde5e6dd784726ef02764c053f040000000000000007000000000000000400000000000000000000000100000001000000780000000000001040ffffffff000000000101000000000000000000000001ed69c16ab099f13e000000000000000000000000000001000000000000005b5344792bba013f0000000000000000000000000001000000000000000100000001000000780000000000000000ffffffff000000000101000000000000000000d03f011e57379fb9d5d33e000000000000020000000000000003000000000000005542bc1dfd09f03e0000000000000000000000000002000000000000000100000001000000780000000000000040ffffffff000000000101000000000000000000e03f01749c94afefb6c73e00000000000004000000000000000500000000000000f7ece98563dfe33e0000000000000000000000000003000000000000000100000001000000780000000000000000ffffffff00000000010000000000000000000000000000000000000000000000000000000600000000000000ffffffffffffffffc30c41993c21d83e000000000000000000000000000000000000000000ea62d91a7b3f7b9c"
    ),
    "offgrid-rechecksummed.n4mopt": bytes.fromhex(
        "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000bd0d7ae11a381dfd"
    ),
    "offgrid-stale-checksum.n4mopt": bytes.fromhex(
        "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90"
    ),
}


@pytest.mark.parametrize(
    "name", ["offgrid-rechecksummed.n4mopt", "offgrid-stale-checksum.n4mopt"]
)
def test_loader_grid_retained_negative(name):
    grid_assert_corrupt_null(LOADER_RETAINED[name])


@pytest.mark.parametrize("name", ["valid.n4mopt", "current-generated.n4mopt"])
def test_loader_grid_retained_positive(name):
    a = Optimizer.load(LOADER_RETAINED[name])
    b = Optimizer.load(LOADER_RETAINED[name])
    try:
        assert a.configuration_matches(b) and history(a) == history(b)
        for record in a.get_trials():
            if record.status == TrialStatus.RUNNING:
                a.tell(record.id, 0.5)
                b.tell(record.id, 0.5)
        for j in range(6):
            x, y = a.ask(), b.ask()
            assert x.id == y.id and bits(x.get_float("x")) == bits(y.get_float("x"))
            a.tell(x.id, j / 8)
            b.tell(y.id, j / 8)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("low,high,step,_expected", SCALAR_CONTROL_VECTORS)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_loader_grid_control_queued(sampler, low, high, step, _expected):
    test_t4f_queued_checkpoint(sampler, low, high, step)


@pytest.mark.parametrize("k", [5, 13, 17])
def test_loader_grid_generated_tolerance_aligned_high(k):
    high = math.nextafter(k * 0.1, -math.inf)
    space = SearchSpace().add_float("x", 0, high, 0.1)
    opt = Optimizer(space, sampler=Sampler.RANDOM, seed=19)
    try:
        found = False
        for _ in range(512):
            trial = opt.ask()
            found = bits(trial.get_float("x")) == bits(high)
            opt.tell(trial.id, 0.25)
            if found:
                break
        assert found
        copy = Optimizer.load(opt.save())
        try:
            assert copy.configuration_matches(opt) and history(copy) == history(opt)
            for j in range(6):
                a, b = opt.ask(), copy.ask()
                assert a.id == b.id and bits(a.get_float("x")) == bits(b.get_float("x"))
                opt.tell(a.id, j / 8)
                copy.tell(b.id, j / 8)
        finally:
            copy.close()
    finally:
        opt.close()
        space.close()
