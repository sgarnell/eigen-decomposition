# Phase 03 — Eigen / SVD Decomposition

*Implementation: `src/spectral/spectral_decomposition.py` (with `src/spectral/__init__.py`).*
*Artifacts: `data/processed/<stem>/eigen.json`, `eigen.npz`, `eigen.png`, `eigen.modes.png`, `eigen.data.json`.*
*Tests: `tests/test_spectral_decomposition.py`, `tests/test_spectral_schema.py`.*

---

## 1. Goals

Extract the **latent functional modes** of the Phase 02 unified-weight matrix `Z` and
persist them as the Phase 03 artifact that Phases 04–06 consume.

Phase 03 owns exactly two decompositions:

* **eigen** — `scipy.linalg.eigh` on a *symmetric* matrix (`Z_sym = (Z + Zᵀ)/2`, or Phase 02's
  `matrix_symmetric`): real eigenvalues, orthonormal eigenvectors.
* **svd** — `scipy.linalg.svd` (`gesdd`, `full_matrices=False`) on the *directed* `Z`:
  non-negative singular values plus the two orthonormal loading bases `U` (source/sender
  axis) and `V` (target/receiver axis) — the "cross-type coupling axes" of the master plan.

`--method auto` (default) picks the eigen path when the selected source matrix is
numerically symmetric and SVD otherwise, so a canonical Phase 02 artifact decomposes with
SVD and a `--symmetric` one with eigen, without extra flags. `--method eigen` on an
asymmetric source is a hard error.

Phase 03 performs **no clustering, no embeddings, no dynamical modelling and no
dimensionality reduction of the data** — the heuristics only *recommend* a mode count.

**Purely additive.** No existing file is modified or deleted: `src/utils/io.py`,
`src/parsing/*`, `src/matrices/*`, `pytest.ini`, `environment.yml`, the Phase 01/02 tests and
the existing `parsed_graph.json` / `z_matrix*` artifacts stay byte-identical. **No new
dependency** (numpy + scipy + matplotlib are already pinned; the elbow detector is
implemented in-house rather than adding `kneed`).

Pipeline order is fixed and recorded in every artifact:

```
select source (effective | symmetric)  ->  decompose (eigh | svd)  ->  rank  ->  sign-fix  ->  retain k
```

## 2. Scope boundary

| Owned by Phase 03 | Explicitly **not** Phase 03 |
|---|---|
| eigen / SVD, ranking, sign convention, retained-`k` slicing | pre/post unification, matrix assembly (Phase 02) |
| the three dimensionality heuristics + participation ratio | pair filtering (Phase 01), clustering (Phase 04) |
| the five artifacts, `--plot`, `--stats`, `--save-data` | reduced models, Jacobians, stability (Phases 05–06) |
| `load_spectrum()` / `load_sidecar_arrays()` for Phases 04+ | multi-file union (never implied) |

## 3. Artifact naming and lineage

Canonical stem **`eigen`**, landing next to the Phase 02 artifacts in
`data/processed/<gv-stem>/`:

| Input (`z_matrix…`) | Phase 03 outputs |
|---|---|
| `z_matrix.json` | `eigen.json`, `eigen.npz`, `eigen.png`, `eigen.modes.png`, `eigen.data.json` |
| `z_matrix.f8652585.json` | `eigen.f8652585.*` |
| `z_matrix.filtered.json` | `eigen.filtered.*` |
| non-default Phase 03 config | inserts `.<config_hash8>` → `eigen.<hash8>.*`, `eigen.f8652585.<hash8>.*` |

* `input_variant()` = the input stem with the leading `z_matrix` stripped (`.f8652585`,
  `.filtered`, or `""`), **inherited** so a spectrum computed from a variant matrix can never
  overwrite the canonical `eigen.json`.
* `variant_stem()` = `eigen` + input variant + (`force_config_hash` or non-default config →
  `.<config_hash8>`).
* `artifact_paths()` = the five names under `<outdir>/<source_file stem>/`.

## 4. `eigen.json` schema

```json
{
  "provenance": {
    "source_artifact": "data/processed/<stem>/z_matrix.json",
    "source_artifact_sha256": "<sha256 of the Phase 02 JSON bytes>",
    "source_file": "<gv stem>.gv",
    "source_file_sha256": "c44c24f85756ed07acc1184565b19407df6fa4e0df8493c761f7d84957fa6b57",
    "parsed_created_utc": "2026-09-17T04:38:36Z",
    "z_matrix_config_hash": "1aa02137",
    "z_matrix_config": { "...": "the Phase 02 config block, verbatim" },
    "phase": "03"
  },
  "config": {
    "method": "auto", "resolved_method": "svd", "source": "effective",
    "rank_by": "magnitude", "sign_convention": "max-abs-positive",
    "k_requested": "auto", "k_resolved": 21,
    "variance_threshold": 0.9, "elbow": true, "elbow_curve": "cumulative",
    "elbow_method": "l-method", "spectral_gap": true, "gap_window": 25,
    "gap_metric": "ratio", "driver": "evr", "residual_tol": 1e-09,
    "degenerate_tol": 1e-10, "symmetric_input": false, "config_hash": "xxxxxxxx"
  },
  "neuron_order": ["DNa03_R_1", "..."],
  "values": [
    {"rank": 1, "original_index": 0, "value": 4.4417728428518295, "abs_value": 4.4417728428518295,
     "energy": 19.729344576539947, "explained_variance_ratio": 0.30782283556009055,
     "cumulative_variance_ratio": 0.30782283556009055}
  ],
  "modes": [
    {"rank": 1, "original_index": 0, "value": 4.4417728428518295, "abs_value": 4.4417728428518295,
     "energy": 19.729344576539947, "explained_variance_ratio": 0.30782283556009055,
     "cumulative_variance_ratio": 0.30782283556009055,
     "left": ["..."], "right": ["..."], "left_norm": 1.0, "right_norm": 1.0,
     "sign_flipped": true, "residual": 1.7e-15}
  ],
  "heuristics": {
    "variance_threshold": {"k": 22, "threshold": 0.9},
    "elbow": {"k": 21, "curve": "cumulative", "method": "l-method"},
    "spectral_gap": {"k": 1, "window": 25, "metric": "ratio", "value": 1.6758602258009703},
    "participation_ratio": 35.6323231094758,
    "recommended_k": 21,
    "rule": "median of the enabled heuristic k values, clipped to [1, N]"
  },
  "metadata": { "...": "see §5" }
}
```

**Deliberate design points**

* `values` carries **every** mode (all `N`), ranked per `config.rank_by`, each with its
  `original_index` (the solver column it came from) — the ranked spectrum is never truncated
  and the ranking is fully traceable.
* `modes` carries only the **retained** `k` loading pairs, so the JSON stays small while
  Phase 04 gets a ready `(N × k)` matrix. `loadings_left`/`loadings_right` are **properties**
  derived from `modes`, never duplicated keys.
* `left` is the source/sender axis (`U` for SVD, `V` for eigen); `right` is the
  target/receiver axis (`V` for both). The eigen path uses the same vector twice, so
  `left == right` there and Phase 04 can read either uniformly.
* **Explained variance is `value² / Σvalue²` for both paths** (classical `σ²/Σσ²` for SVD,
  `λ²/Σλ²` for a symmetric matrix), which makes the two spectra comparable and yields the
  exact identity `Σenergy == ‖A‖_F²`.
* `energy = value²`; `explained_variance_ratio = energy/Σenergy`; `cumulative` is its running
  sum in rank order.
* Writing rules identical to Phase 02: `write_json_atomic` + `dumps_json` (sorted keys,
  indent 2, trailing newline, UTF-8, temp file + `os.replace`), every emitted float asserted
  finite, `-0.0` canonicalized, `created_utc` honours `SOURCE_DATE_EPOCH`.
* `neuron_order` is a **list**, not a map (`dumps_json` sorts object keys, which would destroy
  the index order).
* **No CSV files** are produced or consumed anywhere.

## 5. `metadata` keys (exact)

`n_neurons`, `n_modes_total`, `n_modes_retained`, `method`, `source`, `matrix_is_symmetric`,
`n_positive`, `n_negative`, `n_zero`, `value_min`, `value_max`, `abs_value_max`,
`smallest_value`, `sum_values`, `sum_squares`, `frobenius_norm`, `trace`, `numerical_rank`,
`condition_number`, `explained_variance_top1`, `explained_variance_topk`, `cumulative_at_k`,
`participation_ratio`, `reconstruction_error`, `reconstruction_error_abs`,
`orthogonality_error`, `max_mode_residual`, `degenerate_groups`, `n_sign_flipped`,
`heuristics`, `config_hash`, `scipy_version`, `numpy_version`, `generator`, `created_utc`.

**Statistic conventions** (documented because they are easy to get wrong):

* `frobenius_norm` = `numpy.linalg.norm(A)` — deliberately the *same call* Phase 02 uses, so
  the two artifacts agree bit-for-bit; `sum_squares` = `Σvalue²` is the spectral identity
  check (`sum_squares == frobenius_norm²`).
* `abs_value_max` is the Phase 03 spectral radius: it must equal Phase 02's
  `largest_singular_value` (SVD on `Z`) or `symmetrized_spectral_radius` (eigen on
  `matrix_symmetric`).
* `reconstruction_error` is the *relative* `‖A_rec(k) − A‖_F / ‖A‖_F`. With `k == N` it is the
  full-decomposition check and is asserted `< --residual-tol`; with `k < N` it is the
  (informational) truncation error and is always larger.
* `orthogonality_error` = `max |LᵀL − I|` over both loading bases (≈1e-15).
* `max_mode_residual` = `max_i ‖A v_i − λ_i v_i‖∞` (eigen) or
  `max_i max(‖A v_i − σ_i u_i‖∞, ‖Aᵀ u_i − σ_i v_i‖∞)` (SVD) — this is what catches a
  mis-paired singular triple, e.g. an independently flipped `u_i`/`v_i`.
* `numerical_rank` = `#{σ > max(σ)·N·eps}` and `condition_number = σ₁/σ_rank` — **SVD only**
  (`null` for the eigen path, where the analogous quantity is not meaningful).
* `degenerate_groups` = `{"count", "sizes", "tolerance"}` over clusters of `abs_value` equal
  within `degenerate_tol` (relative). Inside such a cluster the *vectors* are defined only up
  to a rotation, which no sign convention can fix.
* `n_sign_flipped` counts the retained pairs flipped by the sign convention.

## 6. `eigen.npz` sidecar

A **derived cache**, never a second source of truth: its content is a pure function of the
canonical JSON bytes and it is keyed by that JSON's SHA-256. Nothing in the JSON refers to the
cache, so the §4 schema and JSON determinism are untouched.

| Key | Shape/dtype | Meaning |
|---|---|---|
| `values` | `(N,)` `<f8` | the full ranked spectrum |
| `loadings_left` | `(N, k)` `<f8` | `U` (SVD) / `V` (eigen) — the Phase 04 matrix |
| `loadings_right` | `(N, k)` `<f8` | `V` (both paths) |
| `explained_variance_ratio` | `(N,)` `<f8` | ranked |
| `rank_indices` | `(N,)` `<i8` | rank → original solver index |
| `neuron_order` | `(N,)` `<U*` | row/column index order |
| `source_json_sha256` | `(1,)` `<U64` | digest of the JSON this cache derives from |
| `numpy_version` | `(1,)` `<U64` | the numpy that wrote it (`.npy` header is part of the bytes) |

* **Byte-reproducible**: numpy pins every zip entry to the constant `date_time =
  (1980, 1, 1, 0, 0, 0)` and the `.npy` header carries no timestamp (asserted by
  `test_sidecar_zip_entries_use_a_fixed_timestamp`).
* **Atomic**: built in memory (`io.BytesIO`) then written to a temp file in the destination
  directory, flushed, `fsync`-ed and `os.replace`-d — a reader never sees a partial cache.
* **Unpickled**: everything is read with `allow_pickle=False`.
* **Staleness**: `load_spectrum` verifies the digest plus exact array equality; a stale or
  tampered cache is ignored with a warning and the JSON wins. Reading a tampered `.npz`
  directly raises `SpectralValidationError`.
* Writes are ordered JSON first, then the cache, so a failed cache write can never lose the
  artifact of record.

## 7. `eigen.data.json` (`--save-data`)

Keys (exact): `provenance`, `config`, `method`, `source`, `values`, `modes`, `heuristics`,
`n_modes_total`, `n_modes_retained`, `reconstruction_error`, `reconstruction_error_abs`,
`orthogonality_error`, `max_mode_residual`, `energy_share`, `generator`, `created_utc`.

It adds the per-mode detail the canonical JSON deliberately omits: `energy_share`,
`left_argmax`/`right_argmax` (which neuron dominates each axis), `left_stats`/`right_stats`
(`min/max/mean/std` per loading vector) and the per-pair `residual`.

## 8. Algorithm specification

### 8.1 Source selection and method resolution
1. `--source effective` (default) → Phase 02's effective matrix (`Z_sym` if that run used
   `--symmetric`, else the directed `Z`).
2. `--source symmetric` → `matrix_symmetric` (always bit-exactly symmetric).
3. `--method auto` → `eigen` when the selected matrix is numerically symmetric
   (`np.array_equal(A, Aᵀ)` first, then `allclose(rtol=0, atol=1e-12·max(|A|,1))`), else `svd`.
4. `--method eigen` on an asymmetric source → `CODE_METHOD_INCOMPATIBLE` error naming the fix
   (`--method svd`, `--source symmetric`, or rebuild with `--symmetric`).
5. `--source symmetric --method svd` is allowed and is a built-in cross-check (`σ == |λ|` for
   a symmetric matrix — asserted on the reference data).

### 8.2 Decompositions (SciPy first — the requirement)
* **eigen** — `scipy.linalg.eigh(A, lower=True, driver=cfg.driver, check_finite=True)`;
  drivers `evr` (default), `evd`, `ev`, `evx`. Results are real and orthonormal.
* **svd** — `scipy.linalg.svd(A, full_matrices=False, compute_uv=True,
  lapack_driver="gesdd", check_finite=True)`; `U`, `σ`, `V` from `Vt.T`.
  A large-`N` truncated solver (`scipy.sparse.linalg.svds`) was considered and **deliberately
  not implemented**: it cannot produce the complete ranked spectrum §4 promises, so it is
  documented as a future optimization instead of a half-true CLI flag.
* **Stability guards**: the matrix is checked square and finite *before* the call (a NaN would
  otherwise make `allclose` warn and silently yield a `NaN` spectrum); `check_finite=True` is
  passed anyway; afterwards the residuals of §5 are computed, and a full decomposition
  (`k == N`) must reconstruct to within `--residual-tol`.

### 8.3 Ranking, sign convention, retention
* `--rank-by magnitude` (default) = descending `|value|`; `--rank-by value` = descending signed
  value. Ties break on the signed value, then on the original solver index, so the ranking is a
  pure function of the spectrum. `rank_indices` records the mapping in both directions.
* `--sign-convention max-abs-positive` (default): for each retained pair, if the largest-|.|
  component of the **left** vector is negative (first index wins on ties), negate **both**
  vectors of the pair. The coupling is essential: an SVD triple satisfies `A v_i = σ_i u_i`, so
  flipping only one of `u_i`/`v_i` breaks the relation and inflates every per-mode residual
  (observed during implementation; now regression-tested). For the eigen path `left == right`,
  so the coupled flip is exactly a flip of the eigenvector.
  `--sign-convention none` leaves the solver output untouched (diagnostics only) and is
  reported as non-reproducible.
* `--k auto` (default) retains `recommended_k` modes; `--k 0`/`all` retains every mode;
  `--k M` retains `min(M, N)` (clipped to `[1, N]`). All `N` values are always stored.

### 8.4 Dimensionality heuristics (all advisory)
| Heuristic | CLI | Definition | Reference (directed `Z`) |
|---|---|---|---|
| Variance threshold | `--variance-threshold T` (0.90) | smallest `k` with `Σ_{i≤k} r_i ≥ T` | **k = 22** |
| Elbow | `--elbow`/`--no-elbow`, `--elbow-curve {cumulative,scree}`, `--elbow-method {l-method,second-difference}` | L-method: farthest point from the chord joining the curve's endpoints (1-based); alternative: largest absolute second difference | **k = 21** (cumulative) |
| Spectral gap | `--spectral-gap`/`--no-spectral-gap`, `--gap-window N` (25), `--gap-metric {ratio,gap}` | `argmax` of `\|v_i\|/\|v_{i+1}\|` (or `\|v_i\|−\|v_{i+1}\|`) **inside the leading window** | **k = 1**, ratio 1.6758602258009703 |
| Participation ratio | always reported | `(Σ\|v\|)² / Σv²` | 35.6323231094758 |

`recommended_k` = **median of the enabled heuristics' `k` values**, clipped to `[1, N]`
(`median(22, 21, 1) = 21` on the reference data); with no heuristic enabled it falls back to
`N`. `--k auto` follows it.

**Documented trap (design decision, not a bug):** the spectral-gap search is restricted to the
leading `--gap-window` components on purpose. An unrestricted `argmax` of the consecutive ratio
lands on the numerical-null tail — it returns **k = 109** for the reference directed spectrum,
whose four smallest singular values are ~1e-17 apart — which is a measurement artefact rather
than a mode boundary. `--gap-window` must be `>= 2` and is clamped to `N − 1` internally, so it
never errors on a small matrix.

## 9. Diagnostic plots

matplotlib is imported lazily (inside the plot functions) so importing the module never selects a
backend, and headless detection reuses Phase 02's fixed `can_popup()` (matplotlib backend
registry, exact-name fallback for older releases).

* **`eigen.png`** (`--plot`, 150 dpi, three panels):
  * scree — explained-variance ratio vs rank, **log-y** (the reference spectrum spans ~17
    decades, so a linear axis shows nothing after mode 1);
  * cumulative variance with the threshold line and vertical markers for the elbow k, the
    spectral-gap k and the retained k;
  * the spectrum itself — a signed eigenvalue stem plot (eigen) or singular values on a log axis
    with the numerical-rank marker (SVD).
  This is the **only** figure that pops up; `--no-popup` suppresses it and the PNG is written
  either way.
* **`eigen.modes.png`** (`--plot --plot-modes N`, `N = 0` disables): the `(N × k)` loading matrix
  as a heatmap (rows = neuron order, columns = mode rank) plus the top-`N` single-mode
  contributions `value_i · u_i v_iᵀ` (eigen: `λ_i v_i v_iᵀ`) — the "mode heatmaps" of the
  requirements and the cross-type coupling axes. Always saved **silently** (no window), so at
  most one window opens per input.

`--stats` and the completion box are printed to **stdout** and are entirely separate from the GUI
windows.

## 10. Terminal output

`--stats` block (terminal only):

```
Phase 03 -- spectral statistics
  method / source     : svd / effective  (symmetric input: False)
  shape               : 113 x 113
  values              : 113 total, 21 mode(s) retained
  top 5               : 4.441773, 2.650444, 2.370782, 2.324290, 1.830950
  |value| max         : 4.441773
  explained var top1  : 0.307823
  cumulative @ k      : 0.896063
  heuristics          : threshold k=22 | elbow k=21 | gap k=1
  recommended k       : 21
  participation ratio : 35.632323
  frobenius norm      : 8.005822
  sum(value^2)        : 64.093185
  reconstruction error: 3.224e-01  (relative, retained modes)
  orthogonality error : 1.776e-15
  max mode residual   : 1.887e-15
  numerical rank      : 109   condition number: 4.091e+05
  degenerate clusters : 1
  sign flips          : 11   trace: 0.000000
  scipy / numpy       : 1.17.1 / 2.4.6
```

Post-run summary box (always printed on success; terminal only):

```
+--------------------------------------------------------------+
| Phase 03 complete                                            |
| Method: SVD (source: effective)                              |
| Spectrum: 113 values, top |v| = 4.441773                     |
| Modes retained: 21 (recommended_k=21)                        |
| Heuristics: threshold k=22 | elbow k=21 | gap k=1            |
| Cum. variance @k: 0.896063                                   |
| Reconstruction error: 3.224e-01 (relative)                   |
| JSON written: eigen.json                                     |
| NPZ written:  eigen.npz                                      |
| Plot:        not requested (--plot)                          |
| Save-data:   not requested (--save-data)                     |
| Stats: not requested (--stats)                               |
+--------------------------------------------------------------+
```

The box reflects the real outcome (`skipped (--no-sidecar)`, `skipped (--dry-run)`).

## 11. Module layout and data structures

```
src/spectral/__init__.py               # lazy __getattr__ re-exports (Phase 01/02 pattern)
src/spectral/spectral_decomposition.py # engine, heuristics, artifacts, plots, CLI
tests/test_spectral_decomposition.py   # behaviour, artifacts, CLI, invariants
tests/test_spectral_schema.py          # schema + artifact-contract tests
```

Unchanged: `src/utils/io.py`, `src/parsing/*`, `src/matrices/*`, `pytest.ini`, `environment.yml`
and every Phase 01/02 test and artifact.

```python
@dataclass(frozen=True)
class SpectralConfig:      # method, resolved_method, source, rank_by, sign_convention,
                           # k_requested, k_resolved, variance_threshold, elbow, elbow_curve,
                           # elbow_method, spectral_gap, gap_window, gap_metric, driver,
                           # residual_tol, degenerate_tol, symmetric_input
    def to_dict() / from_dict() / hash_fields() / config_hash() / is_default()

@dataclass(frozen=True)
class SpectralValue:       # rank, original_index, value, abs_value, energy, evr, cumulative
@dataclass(frozen=True)
class SpectralMode:        # the value fields + left, right, left_norm, right_norm,
                           # sign_flipped, residual
@dataclass(frozen=True)
class HeuristicResult:     # name, k, value, threshold, curve, method, window, metric
@dataclass(frozen=True)
class DecompositionResult: # method, symmetric_input, n_neurons, k, values, rank_indices,
                           # abs_values, energy, evr, cumulative, left, right,
                           # sign_flipped, numerical_rank
@dataclass
class SpectralDecomposition:
    provenance fields / config / neuron_order / values / modes / heuristics / metadata
    @property n_neurons, n_modes, k, index, spectrum, abs_spectrum,
              explained_variance_ratio, cumulative_variance_ratio,
              loadings_left, loadings_right
    def to_dict()
class SpectralValidationError(ValueError):   # carries .report
```

`metadata` holds the JSON-serializable summaries (the same dict drives `--stats` and
`--save-data`); `loaded_from` records where an object was read from and is never serialized, so
`to_dict()` round-trips byte-for-byte. Phase 01's `ValidationReport`/`ValidationIssue` collection
and the `...Error(ValueError)`-with-`.report` convention are reused verbatim.

`SpectralConfig.hash_fields()` covers `method`, `source`, `rank_by`, `sign_convention`,
`k_requested`, `variance_threshold`, `elbow`, `elbow_curve`, `elbow_method`, `spectral_gap`,
`gap_window`, `gap_metric`, `driver`, `residual_tol`. Derived/observational fields
(`resolved_method`, `k_resolved`, `symmetric_input`, `degenerate_tol`) are recorded but **not
hashed**, so a canonical run still produces the canonical `eigen.json`.

## 12. CLI reference

```
python -m src.spectral.spectral_decomposition \
  -i data/processed/<stem>/z_matrix.json -o data/processed \
  [--glob 'z_matrix.json'] [--include-variants] \
  [--method {auto,eigen,svd}] [--source {effective,symmetric}] \
  [--k auto|0|N] [--rank-by {magnitude,value}] \
  [--sign-convention {max-abs-positive,none}] [--driver {evr,evd,ev,evx}] \
  [--variance-threshold 0.9] [--elbow] [--no-elbow] \
  [--elbow-curve {cumulative,scree}] [--elbow-method {l-method,second-difference}] \
  [--spectral-gap] [--no-spectral-gap] [--gap-window 25] [--gap-metric {ratio,gap}] \
  [--residual-tol 1e-9] \
  [--config-hash] [--no-sidecar] \
  [--plot] [--no-popup] [--cmap viridis] [--plot-modes 4] \
  [--save-data] [--stats] \
  [--strict] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

Exit codes mirror Phase 02: `0` success, `1` any input failed validation/decomposition or an
artifact could not be written, `2` no input matched. `--include-variants` also resolves
`z_matrix.<variant>.json`; directory scans pick up **only** Phase 02 artifacts, so a Phase 03
output or a `z_matrix.data.json` is never re-consumed.

Example commands:

```bash
# canonical spectrum of the directed matrix (SVD, auto k)
python -m src.spectral.spectral_decomposition \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/z_matrix.json -o data/processed --stats

# true eigen-decomposition of the symmetrized matrix
python -m src.spectral.spectral_decomposition \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/z_matrix.json -o data/processed \
  --source symmetric --method eigen --stats --plot --no-popup

# every mode, plot plus mode heatmaps, save-data, hashed filenames
python -m src.spectral.spectral_decomposition \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/z_matrix.json -o data/processed \
  --k 0 --plot --plot-modes 6 --save-data --config-hash

# batch over the whole processed tree, JSON-only, strict
python -m src.spectral.spectral_decomposition -i data/processed -o data/processed \
  --include-variants --no-sidecar --strict --no-popup

# validate without writing anything
python -m src.spectral.spectral_decomposition -i <artifact> -o data/processed --dry-run --stats
```

## 13. Validation rules

| Check | Severity |
|---|---|
| Phase 02 input payload conforms to its schema (reuses `validate_z_payload_schema`) | Error |
| The selected matrix is square and finite (checked before the solver) | Error |
| `--method eigen` requires a numerically symmetric source | Error |
| Valid `method`/`source`/`rank_by`/`sign_convention`/`driver`/`elbow_*`/`gap_metric` | Error |
| `--k` is `auto`/`all`/`0` or an integer in `[0, N]` | Error |
| `variance_threshold ≤ 1`, `gap_window ≥ 2`, `residual_tol > 0`, `degenerate_tol > 0` | Error |
| All emitted floats finite; `-0.0` never appears | Error |
| Emitted payload conforms to the Phase 03 schema | Error |
| Orthonormality error small; `Σvalue² == ‖A‖_F²` | Error |
| Full decomposition (`k == N`) reconstructs to within `--residual-tol` | Error |
| Sidecar / plot / save-data write failure (never a partial file) | Error |
| `N < 2` (`small_graph`) | Warning (`--strict` ⇒ error) |
| All-zero matrix (`trivial_matrix`) | Warning (`--strict` ⇒ error) |
| Numerical-null singular values (`SVD rank < N`) | Informational log line |
| Degenerate spectral cluster(s) (`degenerate_groups.count > 0`) | Informational log line |
| Truncated decomposition (`k < N`) — the truncation error is reported | Informational |
| Popup skipped (headless / `--no-popup`) | Informational log line |

**Why rank deficiency and degeneracy are informational, not warnings** (a deliberate deviation
from the first draft of this plan, justified by Phase 02's precedent for its filtered-input and
popup notices): the reference directed matrix *always* has four numerical-null singular values,
so making them warnings would make `--strict` permanently fail on the canonical dataset while
adding no signal. Both quantities are still recorded in `metadata.numerical_rank` /
`metadata.degenerate_groups` and printed by `--stats`, and the code constants
`CODE_DIMENSION`/`CODE_DEGENERATE` label the log lines.

## 14. Invariants (asserted in tests)

1. eigen: `VᵀV == I` (atol 1e-12); `values` are real; `‖V diag(λ) Vᵀ − A‖_F/‖A‖_F < 1e-12`;
   `Σλ ≈ trace(A)`.
2. svd: `UᵀU == VᵀV == I`; `‖U Σ Vᵀ − A‖_F/‖A‖_F < 1e-12`; `Σσ² ≈ ‖A‖_F²`.
3. **Path agreement on a symmetric input:** `sorted(|λ|) == sorted(σ)` and `Σλ² == Σσ²`
   (rtol 1e-12) — exact on the reference `matrix_symmetric`.
4. `explained_variance_ratio` sums to 1; `cumulative` is monotone and ends at 1;
   `evr == energy/Σenergy`.
5. Every retained `left`/`right` vector has L2 norm 1 **and** its leading component is positive
   (after `max-abs-positive`).
6. `modes[i].rank == i+1`; `rank_indices` is a permutation of `range(N)`; `values` are ordered
   per `rank_by`.
7. `reconstruction_error < residual_tol` whenever `k == N`, and the truncated error is larger.
8. `max_mode_residual < 1e-12` and `orthogonality_error < 1e-12` for both paths.
9. `eigen.json` and `eigen.npz` are byte-identical across runs under a fixed
   `SOURCE_DATE_EPOCH`; `load_spectrum(path).to_dict()` reproduces the artifact byte-for-byte.
10. The npz digest matches the JSON bytes and is a pure function of them; `--no-sidecar` leaves
    one artifact and `load_spectrum()` still works.
11. The Phase 02 object and its payload are **never mutated** by a Phase 03 run.
12. **No `*.csv` is produced anywhere.**

## 15. Reference numbers (verified on the real artifacts)

**Directed `Z`** (`z_matrix.json`, Phase 02 config hash `1aa02137`), `--method auto --source
effective` → **SVD**:

| Quantity | Value |
|---|---|
| σ (top 8) | 4.441772842851828, 2.650444, 2.370782, 2.324290, 1.830950, 1.590538, 1.523123, 1.204741 |
| `abs_value_max` vs Phase 02 `largest_singular_value` | 4.4417728428518295 vs 4.441772842851828 (≈2 ulp: SciPy `gesdd` vs NumPy) |
| `frobenius_norm` | 8.005821952321897 — **equal to Phase 02's** (same `np.linalg.norm` call) |
| `sum_squares` | 64.09318513227922 (≈ `frobenius_norm²` = 64.09318513227919) |
| variance ratio, first 4 | 0.307823, 0.109604, 0.087694, 0.084289 |
| cumulative @4 | 0.589409 |
| k @ 0.80 / 0.85 / 0.90 / 0.95 / 0.99 | 12 / 16 / **22** / 34 / 57 |
| elbow (L-method, cumulative) | **k = 21** |
| spectral gap (window 25, ratio) | **k = 1**, value 1.6758602258009703 |
| participation ratio | 35.6323231094758 |
| `recommended_k` / retained `k` | **21** / 21 |
| cumulative at k | 0.896063 |
| `numerical_rank` / `condition_number` | 109 / 409126.97892414214 |
| degenerate clusters | 1 group of size 4 (the numerical null space) |
| `orthogonality_error` / `max_mode_residual` | 1.78e-15 / 1.89e-15 |
| reconstruction error (k = 21 of 113) | 0.322393 (truncation, informational) |
| sign flips | 11 |

**`matrix_symmetric`** (same artifact), `--source symmetric` → **eigen**:

| Quantity | Value |
|---|---|
| λ, ranked by magnitude (top 5) | −2.433314, 1.768481, 1.100301, −1.094564, −1.028814 |
| `abs_value_max` | 2.4333138382394823 → **equals Phase 02's `symmetrized_spectral_radius`** |
| `sum_squares` | 24.059592651906495 = `∑λ² = ‖Z_sym‖_F²` |
| `trace` | −8.9e-16 (≈0: the diagonal is exactly zero) |
| heuristics | threshold 30 / elbow 25 / gap 2 → retained k = 25 |
| `numerical_rank` / `condition_number` | `null` / `null` (eigen path) |

**`z_matrix.f8652585.json`** (Phase 02 symmetric + unit), `--method auto --source effective` →
**eigen** (its `matrix` is bit-exactly symmetric):

| Quantity | Value |
|---|---|
| `abs_value_max` | 4.317409128240158 → **equals that artifact's `spectral_radius`** |
| `matrix_is_symmetric` | true (hence `auto` picks eigen) |
| heuristics / retained k | threshold 30 / elbow 25 / gap 2 → 25 |

## 16. Determinism

* `created_utc` is the only non-deterministic field and derives from `SOURCE_DATE_EPOCH`
  (verified: repeated runs give identical `eigen.json` and `eigen.npz` sha256).
* Eigenvector **signs** are the other classic source of run-to-run ambiguity and are fixed by
  `--sign-convention max-abs-positive`; `--rank-by` and the tie-break rules fix the ordering.
* Degenerate clusters are the one thing a convention *cannot* fix (the subspace rotation is
  arbitrary); they are detected and reported.
* `scipy_version` and `numpy_version` are recorded in `metadata` because the `.npy` header is part
  of the sidecar's byte layout. Cross-platform bit-identity of `eigh`/`svd` is **not** claimed.
* The sidecar's zip entries use numpy's frozen `date_time` (1980-01-01).

## 17. Testing strategy

**Framework**: `pytest` 9.1.1 with the existing `pytest.ini` (`pythonpath = .`,
`--strict-markers`, the registered `slow` marker — no new markers, no plugin assumptions). Both
test modules pin `matplotlib.use("Agg")` so no test can block on a window.

```
fast   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -m "not slow" -q
full   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -q
```

**Baseline**: Phase 01 + Phase 02 = 268 tests (261 fast + 7 slow), unchanged. Phase 03 adds
**129** (122 fast + 7 slow): `tests/test_spectral_decomposition.py` 106 and
`tests/test_spectral_schema.py` 23. Full suite after the patch: **397 passed** (107 s).

**Test surface**

* *Engine*: NumPy cross-checks for both paths, path agreement on a symmetric input, trace and
  Frobenius identities, orthonormality + reconstruction, `k` slicing/clamping, ranking by
  magnitude vs signed value (with known `rank_indices`), sign-convention positivity,
  **coupling** (the singular relation survives), determinism, all-zero and non-finite inputs.
* *Heuristics*: synthetic spectra with hand-computed answers for each heuristic, disabled
  branches (payload `null`), window clamping, both gap metrics, participation ratio, the median
  rule and `resolve_retained_k` across all accepted `--k` spellings.
* *Builder*: provenance/config/metadata, eigen-on-directed rejection (with the report attached),
  invalid config, malformed Phase 02 payload, non-square/non-finite tampering, all-zero warning,
  `SOURCE_DATE_EPOCH` determinism of *both* artifacts, and that the Phase 02 object is untouched.
* *Artifacts*: naming/variant inheritance/hash behaviour, sidecar bundle + digest + fixed zip
  timestamp + byte reproducibility, round-trip through both JSON and NPZ paths, stale-cache
  fallback with a warning, tampered-cache error, `--no-sidecar`.
* *Plots*: PNG magic bytes + IHDR dimensions for both figures (both branches), popup matrix
  (headless → no window, `--no-popup` → no window, GUI + popup → exactly one window), failure
  reporting.
* *Terminal*: `--stats` labels, all four summary-box variants.
* *CLI*: end-to-end with every artifact, `--dry-run` writes nothing, `--no-sidecar`, `--strict`,
  exit codes `0/1/2`, directory + `--include-variants`, `--k` + `--config-hash`, no CSV, and a
  `python -m` subprocess smoke test.
* *Schema*: hand-rolled conformance plus 16 negative cases (missing/extra keys at every level,
  bad ranks, wrong vector lengths, bad heuristic entries, non-object payloads).
* *Integration (`@pytest.mark.slow`)*: every number in §15, path agreement on the reference
  symmetric matrix, the variant artifact, and a full CLI run on the reference dataset.

## 18. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **SVD sign pairs** — flipping `u_i` and `v_i` independently breaks `A v_i = σ_i u_i` (this actually happened in the first implementation: residuals of ~3.2) | the sign convention flips the **pair** together; `max_mode_residual < 1e-12` is asserted for both paths |
| Eigenvector signs are solver-dependent → unstable artifacts | `--sign-convention max-abs-positive`, unit-tested incl. the coupled-pair case |
| Degenerate eigenvalues/singular values: the vectors are only defined up to a subspace rotation | `metadata.degenerate_groups` + an informational log line; `--stats` prints the count |
| Unrestricted spectral gap returns k ≈ 109 (ratio blows up on the ~1e-17 tail) | mandatory `--gap-window` (default 25), clamped to `N−1` |
| A degenerate/near-zero matrix makes relatives undefined | explicit finiteness + square checks **before** the solver; `trivial_matrix` warning; `frobenius == 0` guarded |
| `--strict` becoming unusable on the reference data (rank deficiency is normal there) | rank deficiency / degeneracy are informational log lines, not warnings (§13) |
| Running `--method eigen` on the directed `Z` (complex spectrum ⇒ meaningless result) | hard `CODE_METHOD_INCOMPATIBLE` error with the fix in the message; `auto` never does it |
| A truncation error that looks like a failure | `k < N` is explicitly informational; the full-decomposition check only applies when `k == N` |
| JSON growing with `k·N` mode vectors | `--k auto` (≈21 modes) by default; `values` is the only full-length block; `loadings_*` are derived, not stored twice |
| Two artifacts drifting into two sources of truth | the cache is strictly derived, keyed by the JSON sha256, verified on load and ignored (with a warning) when stale |
| Cache byte-instability / `allow_pickle` risk | frozen zip `date_time`, fixed-endian dtypes, `allow_pickle=False` everywhere |
| A GUI popup blocking CI | matplotlib pinned to `Agg` in tests; `can_popup()` (Phase 02's fixed registry probe) + `--no-popup`; the PNGs are always written |
| Plotting a 17-decade spectrum on a linear axis | log-y scree + log-y singular-value panel |
| Phase 03 silently consuming its own output or a save-data file | directory resolution accepts only `z_matrix[.variant].json`; `*.data.json` is explicitly rejected |
| Future large-`N` matrices | full `gesdd`/`eigh` at `N = 113` is microseconds; a truncated `svds` path is documented as a future optimization that must still produce the full ranked spectrum |
| NumPy version drift changing the `.npy` bytes | `scipy_version`/`numpy_version` are recorded; the sidecar is a cache, not the artifact of record |

## 19. Acceptance criteria

1. `--method auto --source effective` on the canonical `z_matrix.json` runs **SVD** and reproduces
   every number in §15 with zero errors and zero warnings.
2. `--source symmetric` (or the `f8652585` variant with `auto`) runs **eigen** and reproduces its
   reference numbers, including `abs_value_max ==` Phase 02's spectral radius.
3. Artifacts: `eigen.json` always; `eigen.npz` unless `--no-sidecar`; `eigen.png` with `--plot`;
   `eigen.modes.png` with `--plot --plot-modes > 0`; `eigen.data.json` with `--save-data` —
   **no CSV**.
4. Naming: the input variant suffix is inherited; a non-default config (or `--config-hash`) adds
   `.<config_hash8>`; the canonical artifact is never overwritten by a variant run.
5. The three heuristics are reported, `recommended_k` follows the documented median rule, and
   `--k` overrides it (`0`/`all` = every mode).
6. `--plot` writes both valid PNGs and opens at most **one** window; `--no-popup` and headless
   backends suppress it; `--stats` prints the block plus the box (terminal only).
7. `--strict` fails on any error-level finding and passes on the reference dataset; exit codes are
   `0/1/2`.
8. `eigen.json` and `eigen.npz` are byte-stable under a fixed `SOURCE_DATE_EPOCH`;
   `load_spectrum()`/`load_sidecar_arrays()` round-trip; the cache is verified, and a stale or
   tampered cache never corrupts a load.
9. The twelve §14 invariants hold, incl. `Σσ² == Σλ² == ‖A‖_F²` for a symmetric input.
10. **No Phase 01/02 file, test or artifact changes**: `git diff --stat` touches only
    `src/spectral/*`, the two new test modules, `README.md` and this plan. The Phase 01/02 tests
    still pass unchanged.

## 20. Deliverables

- `execution-plans/03_Eigen_Decomposition.md` (this document)
- `src/spectral/__init__.py`, `src/spectral/spectral_decomposition.py`
- `tests/test_spectral_decomposition.py`, `tests/test_spectral_schema.py`
- `README.md` Phase 03 section (inputs/outputs, artifacts, CLI, heuristics, plots, Phase 04
  loading, Status row)
- Artifacts: `data/processed/<stem>/eigen.json`, `eigen.npz` (+ `eigen.png`, `eigen.modes.png`,
  `eigen.data.json` when requested)
- **No changes** to `src/utils/io.py`, `src/parsing/*`, `src/matrices/*`, `pytest.ini`,
  `environment.yml`, the Phase 01/02 tests or the existing Phase 01/02 artifacts

## 21. Out of scope (explicit non-goals)

Clustering, UMAP/PCA embeddings, loading-matrix analysis (Phase 04); reduced dynamical models,
Jacobians, stability (Phases 05–06); matrix construction and filtering (Phases 01–02); multi-file
union; complex/`eigvals` diagnostics for the directed matrix (Phase 03 deliberately routes the
directed case through SVD instead); an `--interactive` window (Phase 02 owns interactive
inspection); truncated/sparse large-`N` solvers (`svds`, `lobpcg`) — documented as a future
optimization; per-mode permutation/gap tests and a formal dimensionality estimator beyond the
three heuristics.

## 22. How Phase 04 consumes these artifacts

```python
from src.spectral.spectral_decomposition import load_sidecar_arrays, load_spectrum

# fast path: arrays only (no JSON parsing, no pickle)
arrays = load_sidecar_arrays("data/processed/<stem>/eigen.npz")   # or pass the .json path
values     = arrays["values"]             # ranked spectrum (length N)
loadings   = arrays["loadings_left"]      # the (N x k) matrix L for clustering
order      = arrays["neuron_order"]       # row index order == Phase 02's neuron_order

# full object: verified against the cache, with config/metadata/heuristics
spectrum = load_spectrum("data/processed/<stem>/eigen.json")
spectrum.config          # method / source / rank_by / config_hash
spectrum.metadata        # numerical_rank, degenerate_groups, reconstruction_error, ...
spectrum.heuristics      # variance_threshold / elbow / spectral_gap / recommended_k
spectrum.index           # neuron_id -> row index
spectrum.loadings_right  # the target-side basis (SVD) for cross-type coupling analysis
```

* Symmetric mode (`config.resolved_method == "eigen"`) gives an orthonormal real eigenbasis, so the
  columns of `L` are orthogonal axes of functional variation.
* Directed mode (`"svd"`) gives `L = U` (source/sender roles) and `loadings_right = V`
  (target/receiver roles) — the two bases Phase 04 can cluster on separately or jointly.
* `metadata.config_hash` ties the spectrum to the exact Phase 02 + Phase 03 recipe, so a Phase 04
  artifact can record which spectrum variant it clustered.
* `metadata.degenerate_groups` warns Phase 04 when some columns are not individually meaningful.

## 23. Flagged decisions and defaults

1. **Artifact stem `eigen` with inherited Phase 02 variant lineage** (confirmed by the user).
2. **`--k auto` by default** (retains `recommended_k` vectors, stores all `N` values); `--k 0` /
   `--k all` keeps every mode.
3. **Heuristics are on by default but purely advisory** — they only annotate metadata and the plot;
   disabling them (`--no-elbow`, `--no-spectral-gap`, `--variance-threshold 0`) is possible, and
   `recommended_k` then falls back to `N`.
4. **In-house L-method elbow**, no `kneed` dependency.
5. **`eigen.modes.png` is saved but never popped up**; only `eigen.png` opens a window, so at most
   one window appears per input.
6. **No `--interactive`** in Phase 03 (Phase 02 owns interactive inspection).
7. **`explained_variance_ratio = value²/Σvalue²` for both paths**, which gives the exact
   `Σenergy == ‖A‖_F²` invariant and comparable spectra.
8. **The `svds`/large-`N` truncated SVD option was dropped** from the first draft: it cannot honour
   the "all `N` values" contract, so it is documented rather than exposed as a flag.
9. **Rank deficiency and degeneracy are informational, not warnings** (§13) so `--strict` stays
   usable on the reference dataset.
10. **The elbow reference is `k = 21`, not `20`** as the pre-implementation probe suggested: that
    probe returned a 0-based index while `l_method_knee()` (correctly) returns a 1-based `k`. The
    corrected values are `elbow = 21`, `recommended_k = 21`, retained `k = 21`.
11. **`--save-data` is included** as an optional extra mirroring Phase 02; the canonical
    `eigen.json` is self-sufficient without it.
12. **This plan is written into the existing `execution-plans/03_Eigen_Decomposition.md`** (the file
    already existed and was empty) rather than creating a second, lowercase plan file.
