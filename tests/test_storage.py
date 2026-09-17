"""Checkpoint planning must account for the buffers actually allocated."""

import pytest

from nami.common.storage import storage_plan


def test_auto_plan_scales_with_snapshot_stream_memory():
    # Doubling both checkpoint and snapshot fields scales memory uniformly;
    # it must not change the selected schedule.
    first = storage_plan(10000, 18, 1, 6, True)
    doubled = storage_plan(10000, 36, 1, 12, True)
    assert first == doubled
    assert first[0] > 0


def test_auto_plan_shortens_interval_when_snapshot_streams_increase():
    few = storage_plan(10000, 18, 1, 1, True)
    many = storage_plan(10000, 18, 1, 24, True)
    assert few[0] > many[0] > 0


@pytest.mark.parametrize("sample_steps", [7, 20], ids=["larger", "equal"])
def test_auto_plan_falls_back_when_full_storage_is_no_larger(sample_steps):
    # Cover both a strictly larger checkpoint plan and an equal-cost plan.
    auto = storage_plan(100, 6, sample_steps, 1, True)
    full = storage_plan(100, 6, sample_steps, 1, True, ckpt_steps=0)
    assert auto == full
    assert auto[1] == []  # No replay when retaining the full snapshot stream.


def test_explicit_interval_only_allocates_for_actual_steps():
    ce, segments, n_snap, n_ckpt = storage_plan(
        100,
        6,
        3,
        1,
        True,
        ckpt_steps=1000000,
    )
    assert ce == 1000000
    assert segments == [(0, 100)]
    assert n_snap == 34
    assert n_ckpt == 0


@pytest.mark.parametrize("n_streams", [0, -1], ids=["zero", "negative"])
def test_n_streams_must_be_positive(n_streams):
    with pytest.raises(ValueError, match="n_streams"):
        storage_plan(100, 6, 1, n_streams, True)
