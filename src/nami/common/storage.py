"""Snapshot storage for FWI adjoints (device-backed).

Snapshots always live on the GPU.  ``SnapshotStorage`` owns one snapshot
stream: an ``[n_snap, n_shots, ny, nx]`` device tensor that the forward
kernels write into and the backward kernels read from.  The C++ extension
owns the buffer; the Python layer only holds a handle.

Public knobs on each propagator (defaults in parentheses):

* ``storage`` (``"auto"``): whether to keep snapshots for backward.
  ``"auto"`` enables them when ``torch.is_grad_enabled()`` and any
  differentiable input requires grad; ``"none"`` is forward-only.
* ``sample_steps`` (``1``): sample snapshots / model gradients every N
  time steps (receiver traces stay exact).  Larger values save memory but
  approximate the model gradient.
* ``ckpt_steps`` (``None``): checkpoint schedule.  ``None`` auto-selects
  a square-root-scale interval from ``nt``, state size, snapshot-stream
  count, and ``sample_steps``, falling back to full storage when checkpoint
  states plus snapshot buffers would not be smaller; ``0`` keeps every
  sampled snapshot (full storage); ``N > 0`` saves full wavefield state
  every N steps and replays the segment on backward.  Snapshot capacity is
  capped by the run length.  Checkpointing does not change gradients
  relative to full storage at the same ``sample_steps``.

``compute_checkpoint_plan`` / ``storage_plan`` turn those knobs into
``(checkpoint_every, segments, n_snap, n_ckpt)`` for the front ends.
"""

import contextlib
import math

import torch


class SnapshotStorage:
    """One snapshot stream, backed by a C++ ``SnapshotStore``."""

    def __init__(self, ext, n_snap, n_shots, ny, nx, dtype, device):
        self.ext = ext
        self.n_snap = n_snap
        self.shot_count = n_shots * ny * nx
        dtype_index = 0 if dtype == torch.float32 else 1
        device_index = device.index if device.index is not None else 0
        self._handle = ext.storage_create(
            n_snap, n_shots, ny, nx, dtype_index, device_index
        )
        self._snap = None

    @property
    def snap(self):
        """The snapshot tensor (``[n_snap, n_shots, ny, nx]``, device)."""
        if self._snap is None:
            self._snap = self.ext.storage_snap_tensor(self._handle)
        return self._snap

    def snap_offset(self, step_idx):
        """Flat element offset of snapshot ``step_idx`` within ``snap``."""
        return int(step_idx) * self.shot_count

    def close(self):
        if self._handle is not None:
            self.ext.storage_destroy(self._handle)
            self._handle = None

    def __del__(self):
        # Handle leaks are worse than a double-free guard here; the object is
        # only destroyed once, so this is safe even after an explicit close().
        with contextlib.suppress(Exception):
            self.close()


def resolve_storage(storage):
    """Validate ``storage`` ('auto' or 'none'); return it unchanged."""
    if storage not in ("auto", "none"):
        raise ValueError(f"storage must be 'auto' or 'none' (got {storage!r}).")
    return storage


def check_sample_steps(sample_steps):
    """Validate ``sample_steps`` (an int >= 1); return it as an int."""
    try:
        sample_steps = int(sample_steps)
    except (TypeError, ValueError):
        raise ValueError("sample_steps must be an integer >= 1.") from None
    if sample_steps < 1:
        raise ValueError("sample_steps must be >= 1.")
    return sample_steps


def _check_n_streams(n_streams):
    """Validate ``n_streams`` (an int >= 1); return it as an int."""
    try:
        n_streams = int(n_streams)
    except (TypeError, ValueError):
        raise ValueError("n_streams must be an integer >= 1.") from None
    if n_streams < 1:
        raise ValueError("n_streams must be >= 1.")
    return n_streams


def _sampled_slots(n_steps, sample_steps):
    """Snapshot slots covering ``n_steps`` time steps at ``sample_steps``."""
    return (n_steps + sample_steps - 1) // sample_steps


def compute_checkpoint_plan(nt, n_state, sample_steps, n_streams):
    """Auto checkpoint plan: ``(checkpoint_every, segments, n_snap)``.

    ``checkpoint_every`` balances the checkpoint-state memory
    (``nt / C`` states of ``n_state`` fields) against the replay snapshot
    memory (``n_streams * C / sample_steps`` fields)::

        C* = round(sqrt(n_state * nt * sample_steps / n_streams))

    clamped to ``nt``.  ``n_snap`` is the snapshot-stream capacity needed
    for one segment, ``ceil(C* / sample_steps)``.
    """
    n_streams = _check_n_streams(n_streams)
    checkpoint_every = max(
        1, min(nt, round(math.sqrt(n_state * nt * sample_steps / n_streams)))
    )
    segments = checkpoint_segments(nt, checkpoint_every)
    n_snap = _sampled_slots(checkpoint_every, sample_steps)
    return checkpoint_every, segments, n_snap


def storage_plan(nt, n_state, sample_steps, n_streams, enabled, ckpt_steps=None):
    """Full storage plan ``(checkpoint_every, segments, n_snap, n_ckpt)``.

    ``enabled=False`` (forward-only) returns ``(0, [], 0, 0)``.
    ``ckpt_steps`` is the public schedule: ``None`` selects the auto plan,
    falling back to full storage if checkpoint and snapshot buffers would
    not be smaller.  ``0`` keeps every sampled snapshot (full storage),
    ``N`` checkpoints the wavefield state every N steps.
    """
    if not enabled:
        return 0, [], 0, 0
    n_streams = _check_n_streams(n_streams)
    if ckpt_steps is None:
        ce, segments, n_snap = compute_checkpoint_plan(
            nt,
            n_state,
            sample_steps,
            n_streams,
        )
        n_ckpt = max(0, (nt - 1) // ce)
        full_n_snap = _sampled_slots(nt, sample_steps)
        if n_state * n_ckpt + n_streams * n_snap >= n_streams * full_n_snap:
            return 0, [], full_n_snap, 0
        return ce, segments, n_snap, n_ckpt
    checkpoint_every = int(ckpt_steps)
    if checkpoint_every < 0:
        raise ValueError("ckpt_steps must be >= 0 (or None for auto).")
    if checkpoint_every == 0:
        return 0, [], _sampled_slots(nt, sample_steps), 0
    segments = checkpoint_segments(nt, checkpoint_every)
    n_snap = _sampled_slots(min(checkpoint_every, nt), sample_steps)
    n_ckpt = max(0, (nt - 1) // checkpoint_every)
    return checkpoint_every, segments, n_snap, n_ckpt


def checkpoint_segments(nt, checkpoint_every):
    """[(s0, s1)] forward segments for ``checkpoint_every``; [] = full storage.

    A checkpointed backward replays each segment's forward pass (from the
    saved wavefield state at ``s0``) before running the adjoint steps, so
    the snapshot stream only needs ``checkpoint_every`` slots.
    """
    if checkpoint_every is None or checkpoint_every <= 0:
        return []
    return [
        (s0, min(s0 + checkpoint_every, nt)) for s0 in range(0, nt, checkpoint_every)
    ]
