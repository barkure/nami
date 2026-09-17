# AGENTS.md

Instructions for coding agents. Installation, public API, and engineering
contracts for this repository live here.

## Platform

- GNU/Linux with CUDA only (no macOS/Windows; no CPU-only kernels)
- Source install only: `nvcc` builds extensions at install or `build_ext` time
- Python ≥ 3.12, CUDA-enabled PyTorch, matching toolkit via `CUDA_HOME`

## Install

```bash
export CUDA_HOME=/usr/local/cuda   # machine-specific
pip install "git+https://github.com/barkure/nami.git"
```

For local development, install the package and development tools into the
active Python environment:

```bash
pip install -e .[dev]
```

After changing `src/nami/csrc/*.cu`, rebuild and sanity-check:

```bash
python setup.py build_ext --inplace
python -c "import torch; assert torch.cuda.is_available(); import nami; print(nami.__version__)"
```

Dev tooling and the full test command are under [Tests](#tests).

## Layout

```text
src/nami/
  common/      # survey, PML, CFL, storage, callbacks/state, FD coefficients
  csrc/        # CUDA sources; listed in setup.py
  scalar/      # acoustic 2D/3D + Born
  elastic/     # elastic 2D + Born
  em/          # EM 2D TM, EM 3D + Born; _common.py shared helpers
  models.py    # nn.Module FWI wrappers
  wavelets.py  # Gaussian/Ricker/Ormsby/Klauder, sine burst, chirp
setup.py       # CUDAExtension list only (metadata in pyproject.toml)
tests/
```

Native extensions use top-level names (`nami_scalar2d`, `nami_em2d_tm_born`, …)
because `TORCH_EXTENSION_NAME` cannot contain dots. The `nami_` prefix avoids
collisions in the process-wide module table. Public imports go through the
Python package, not the extension names. Renames require coordinated updates
to the corresponding `.cu` `PYBIND11_MODULE`, `setup.py`, and Python `import`s.

Extension map (Born):

| Kernel package | Source | Python callers |
|---|---|---|
| `nami_scalar2d_born` | `scalar2d_born.cu` | `scalar2d_born` |
| `nami_scalar3d_born` | `scalar3d_born.cu` | `scalar3d_born` |
| `nami_elastic2d_born` | `elastic2d_born.cu` | `elastic2d_born` |
| `nami_em2d_tm_born` | `em2d_tm_born.cu` | `em2d_tm_born` |
| `nami_em3d_born` | `em3d_born.cu` | `em3d_born` |

## Time-stepping contract

Each forward or adjoint pass runs in one native-extension call. Keep the
per-step extension exports available when extending the propagators.

When changing time loops or storage, preserve ring-buffer ordering,
global-time indexing, exact-adjoint behaviour, checkpoint/full-storage
parity, and existing numerical results. C++ ring-buffer indices must never
use a modulo expression that can produce a negative value. CUDA-specific
implementation contracts live in `src/nami/csrc/AGENTS.md`.

## Public imports

| Capability | Import |
|---|---|
| Acoustic 2D / 3D | `from nami.scalar.scalar2d import scalar2d` / `from nami.scalar.scalar3d import scalar3d` |
| Elastic 2D | `from nami.elastic.elastic2d import elastic2d` |
| EM 2D TM / 3D | `from nami.em.em2d_tm import em2d_tm` / `from nami.em.em3d import em3d` |
| Born | e.g. `from nami.em import em3d_born` (each physics subpackage re-exports its `*_born`) |
| Class API | `from nami import Scalar, Scalar3D, Elastic, TM2D, EM3D` |
| Source waveforms | `from nami.wavelets import chirp, gaussian, gaussian_derivative, klauder, ormsby, ricker, sine_burst` |

## Source waveforms

Waveform helpers return one-dimensional tensors on PyTorch's default device;
move or reshape them for the target survey. `gaussian_derivative` supports
normalized orders 1 and 2,
with order 2 opposite in sign to `ricker`. `ormsby` takes four increasing
corner frequencies, and `sine_burst` produces a finite Hann-tapered burst by
default. `chirp` generates a finite linear sweep; `klauder` generates its
normalized zero-phase autocorrelation. Propagators also accept arbitrary
sampled source amplitudes directly.

## Minimal example (acoustic 2D forward)

```python
import torch
from nami.scalar.scalar2d import scalar2d
from nami.wavelets import ricker

device, ny, nx, nt, grid_spacing, dt = "cuda", 100, 100, 300, 5.0, 1e-3
v = 1500 * torch.ones(ny, nx, device=device)
v[ny // 2 :] = 2000
amp = ricker(25.0, nt, dt, 0.06).reshape(1, 1, -1).to(device)
srcs = torch.tensor([[[10, 10]]], device=device)
recs = torch.tensor([[[10, 90]]], device=device)
out = scalar2d(
    v, grid_spacing, dt,
    source_amplitudes=amp,
    source_locations=srcs,
    receiver_locations=recs,
    accuracy=2,
    pml_width=20,
    pml_freq=25.0,
    storage="none",
)  # out: [nt, n_shots, n_rec]
```

Class API: construct `Scalar(v, grid_spacing, dt, ...)`, then call
`.forward(...)`.
Wrapper constructors accept `storage` / `sample_steps` / `ckpt_steps`;
`EM3D` additionally takes `source_component` / `receiver_component`.
`.forward(...)` accepts `forward_callback` / `callback_frequency` /
`return_state` / `initial_state` and passes them to the underlying
propagator. With `return_state=True`, it returns `(receiver_amplitudes,
state)` and caches the state on `.last_state`. For FWI, enable grad on the
model, use `storage="auto"`, and call `.backward(loss)` (writes `param.grad`).

## Propagator parameters

| Parameter | Default | Role |
|---|---|---|
| `storage` | `"auto"` | `"auto"`: retain GPU snapshots when inputs need grad; `"none"`: forward-only (backward raises) |
| `sample_steps` | `1` | Sample snapshots / model grads every N steps (receivers stay exact). For `N > 1`, model grads use the rectangle rule: each sampled imaging term is scaled by `N` (`scale = float(sample_steps)` into every kernel). `N = 1` is bitwise identical on every path |
| `ckpt_steps` | `None` | `None` selects a square-root-scale interval using `nt`, state size, snapshot-stream count, and `sample_steps`, falling back to full storage when checkpoint and snapshot buffers would not be smaller; `0` full snapshot storage; `N > 0` checkpoint every N steps (same grads as full storage at the same `sample_steps`; snapshot capacity is capped by the actual run length) |
| `accuracy` | `2` | Spatial FD order: 2, 4, 6, or 8 |
| `pml_width` | `20` | Scalar width or per-side list; `0` disables that side |
| `pml_freq` | `25.0` | Acoustic/elastic C-PML design frequency (Hz); unused for EM |
| `nt` | from sources | Time steps when not implied by `source_amplitudes` |
| `forward_callback` | `None` | Called during forward propagation with a `CallbackState`; use `get_wavefield(name, view)` to inspect `"inner"`, `"pml"`, or `"full"` wavefields. Returned wavefields are live views of reused CUDA buffers — `.clone()` to keep a snapshot |
| `callback_frequency` | `1` | Invoke `forward_callback` every N time steps |
| `return_state` | `False` | Include a dict of the final padded wavefield state in the return value; keys are documented by each propagator. State dicts are ephemeral runtime snapshots (same propagator / model layout / nami version only), not a stable checkpoint format |
| `initial_state` | `None` | Restore state for forward continuation: missing keys are zero-filled, while the complete `return_state=True` result makes a split run bitwise match a one-shot run; gradients do not propagate across this boundary |
| `bg_receiver_locations` | `None` | Born-only optional background-field receivers; adds `r_bg` after the scattered gather in the return tuple |

dtype and device follow the input models (CUDA `float32` / `float64`).

## Multi-shot contract

Contract for every full-wave and Born propagator (scalar, elastic, EM).

- **`n_shots`** comes from the survey (`source_amplitudes` /
  `source_locations`). Shared models stay `[spatial]` or `[1, spatial]`;
  wavefields are still `n_shots`-wide.
- Shared/per-shot layout follows each user model independently. Every
  multi-parameter propagator may mix shared and per-shot model tensors.
- Gradients for shared models are reduced across shots after the adjoint;
  gradients for per-shot models retain their shot dimension.
- **Outputs:** receiver gathers are `[nt, n_shots, n_rec]`.

## Physics coverage

Full-wave forward + adjoint and Born scattering for scalar acoustic 2D/3D,
elastic 2D (pressure sources/receivers), and EM 2D TM / 3D. Oversized `dt`
raises a CFL error (`nami.common.cfl`).

## Agent guidelines

**Do**

- Keep changes Linux/CUDA-oriented
- Rebuild extensions after `.cu` edits and run the matching `tests/test_*.py`
- Preserve exact-adjoint, checkpoint-parity, and wavefield-continuation tests
  when touching time loops, storage, or kernels
- Prefer `storage="none"` for forward-only work

**Do not**

- Add CPU-only or non-Linux support unless explicitly requested
- Assume prebuilt wheels or a PyPI release path
- Rename top-level extension modules without a coordinated update of `.cu`,
  `setup.py`, and Python imports

## Tests

```bash
pytest                          # full suite (CUDA required)
pytest tests/test_scalar2d.py   # single module
ruff check
```

The optional checked-in pixi environment provides the same development
commands: run `pixi install -e dev` once, then prefix commands with
`pixi run -e dev` (for example, `pixi run -e dev pytest`). Its CUDA PyTorch
pin is machine-specific and may need adjustment. Ruff configuration lives in
`pyproject.toml`.
