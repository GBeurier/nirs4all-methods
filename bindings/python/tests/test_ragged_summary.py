"""Independent packed-series oracle and explicit absence refusals."""
import numpy as np
import pytest
from n4m.roles import RaggedSummary


def test_ragged_summary_numpy_oracle_with_presence_and_physical_time():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(9, 3))
    offsets = [0, 2, 2, 7, 9]
    presence = [1, 0, 1, 1]
    time = [0., 0.25, 1., 1.5, 2., 4., 6., 8., 9.]
    result = RaggedSummary(offsets=offsets, presence=presence,
        time_coordinates=time, missing_policy="zero_with_indicator").run(X)
    expected = np.zeros((4, 15))
    for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if not presence[i]:
            continue
        rows = X[start:end]
        expected[i, :12] = np.stack([rows.mean(0), rows.std(0), rows.min(0), rows.max(0)], axis=1).reshape(-1)
        expected[i, 12:] = [end-start, time[end-1]-time[start], 1]
    np.testing.assert_allclose(result["features"], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(result["presence"], presence)


@pytest.mark.parametrize("options", [
    {"offsets": [1, 3]}, {"offsets": [0, 2]}, {"offsets": [0, 2, 1, 3]},
    {"offsets": [0, 0, 3]},
    {"offsets": [0, 1, 3], "presence": [0, 1]},
    {"offsets": [0, 0, 3], "presence": [0, 1]},
    {"offsets": [0, 1, 3], "presence": [1]},
    {"offsets": [0, 3], "time_coordinates": [0, 1]},
    {"offsets": [0, 3], "time_coordinates": [0, 1, 1]},
])
def test_incoherent_or_missing_sequences_refused(options):
    with pytest.raises(Exception):
        RaggedSummary(**options).run(np.ones((3, 2)))


def test_present_nonfinite_refused_and_single_observation_duration_zero():
    with pytest.raises(Exception):
        RaggedSummary(offsets=[0, 2]).run(np.array([[1., 2.], [np.nan, 3.]]))
    output = RaggedSummary(offsets=[0, 1]).run(np.array([[2., 5.]]))["features"]
    np.testing.assert_array_equal(output[0, -3:], [1, 0, 1])

@pytest.mark.parametrize("options", [
    {"offsets": [0, 1.8, 3]},
    {"offsets": [0, 3], "presence": [0.5], "missing_policy": "zero_with_indicator"},
    {"offsets": [[0, 3]]},
])
def test_parameter_integer_vectors_refuse_lossy_conversion(options):
    with pytest.raises((ValueError, TypeError)):
        RaggedSummary(**options).run(np.ones((3, 2)))


def test_extreme_finite_statistics_and_duration_overflow_refusal():
    output = RaggedSummary(offsets=[0, 2]).run(np.array([[1.e308], [-1.e308]]))["features"]
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output[0,:4], [0.,1.e308,-1.e308,1.e308], rtol=2e-15)
    with pytest.raises(Exception):
        RaggedSummary(offsets=[0, 2], time_coordinates=[-1.e308, 1.e308]).run(np.ones((2, 1)))
