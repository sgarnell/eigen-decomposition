# Sub-phase 04A — Participation matrix & motif extraction

*Implementation: `src/clustering/functional_motifs.py` (with `src/clustering/__init__.py`).*
*Artifacts: `data/processed/<stem>/motifs.json`, `motifs.npz`, `motifs.data.json`.*
*Tests: `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`.*

---

## 1. Goals

Turn the Phase 03 spectrum (`eigen[.<variant>].json`) into the **per-mode signed motif
structure** and stand up the canonical Phase 04 artifact, CLI and the **frozen** payload
schema that 04B–04D populate one layer at a time.

04A answers exactly one question — *what is each mode made of?* — and owns nothing else.
Pipeline order is fixed and recorded in every artifact:

```
load spectrum -> participation -> thresholds -> membership -> signed members -> statistics
```

**Purely additive.** No existing file is modified or deleted: `src/utils/io.py`,
`src/parsing/*`, `src/matrices/*`, `src/spectral/*`, `pytest.ini`, `environment.yml`, the
Phase 01–03 tests and the existing `parsed_graph.json` / `z_matrix*` / `eigen*` artifacts
stay byte-identical. **No new dependency** (numpy only; no sklearn/networkx/matplotlib).

## 2. Scope boundary

| Owned by 04A | Explicitly **not** 04A |
|---|---|
| participation, thresholds, membership, signed members, motif statistics | link classification / family merging (04B) |
| region derivation, `--anatomy`, `parsed_graph.json` enrichment | functional groups (04C) |
| `motifs.json` + `motifs.npz` + `motifs.data.json`, CLI, schema, validator | pathway graph + figures (04D) |
| `load_motifs()` / `load_sidecar_arrays()` for 04B–04D | any plot, sklearn, networkx, matplotlib |

## 3. Inputs — the Phase 03 → Phase 04 contract

`eigen[.<variant>].json` through `load_spectrum()`:

* `neuron_order` — the row index order (a list, never a map);
* `values` — the ranked spectrum, **all** `N` entries;
* `modes` — the retained `k` loadings, `left` (sender axis) and `right` (receiver axis);
* `config` / `metadata` / `heuristics` — recorded in the provenance.

Optional enrichment, never required:

* the sibling `parsed_graph.json` (Phase 01) — supplies `cent`, `out_degree`, `in_degree`;
* `--anatomy PATH` — a JSON mapping/record list that overrides `derive_region`.

`eigen.data.json`, `eigen.png`, `eigen.modes.png` are rejected by `resolve_inputs`.
**No CSV files are produced or consumed anywhere in this project.**

## 4. Outputs — the artifact set

| Artifact | Written | Purpose |
|---|---|---|
| `motifs.json` | always | canonical, self-sufficient motif structure + provenance + config + metadata |
| `motifs.npz` | unless `--no-sidecar` | derived array cache (strictly a function of the JSON bytes) |
| `motifs.data.json` | `--save-data` | per-neuron participation/recurrence/region/`cent`/degrees tables |

**Naming.** `CANONICAL_STEM = "motifs"`; the Phase 03 variant suffix is **inherited**
(`eigen.json` → `motifs.json`, `eigen.f8652585.json` → `motifs.f8652585.json`) and any
non-default Phase 04 config inserts `.<config_hash8>` (or `--config-hash` forces it), so a
variant run can never clobber the canonical artifact. All artifacts land in
`<outdir>/<source_file stem>/`, next to Phase 01–03.

## 5. Participation — the core definition

`compute_participation(loadings_left, loadings_right, variant)` → `(N, k)` float64, finite and
non-negative. Four variants (`PARTICIPATIONS`):

| `--participation` | definition | meaning |
|---|---|---|
| `left` | `\|L\|` | sender-axis magnitude |
| `right` | `\|R\|` | receiver-axis magnitude |
| `max` | `max(\|L\|, \|R\|)` | strongest axis |
| **`rms`** (default) | `sqrt((L² + R²) / 2)` | root-mean-square of both axes |

`rms` is the default because it is the only variant whose `0.25` relative threshold
reproduces the §14 reference numbers exactly (verified on the real artifact).

**Zero-row detection:** a neuron whose whole participation row is exactly `0.0` is
*isolated* and can never become a member. On the reference data only `DNa03_R_1` is isolated
(`n_isolated_neurons = 1`), which is why `n_unique_members = 107` and
`n_neurons_in_no_motif = 6` (one isolated + five sub-threshold neurons).

## 6. Thresholds, membership and members

`motif_thresholds(participation, method=..., ...)` → `(k,)` effective thresholds. Each method
is a **pure function of exactly one knob** — there is no cross-method floor:

| `--threshold-method` | `t_m` | default knob |
|---|---|---|
| **`relative`** (default) | `relative_threshold · max_i P[i,m]` | `--relative-threshold 0.25` |
| `absolute` | `absolute_threshold` (constant) | `--absolute-threshold 0.05` |
| `quantile` | `quantile(P[:,m], quantile)` | `--quantile 0.8` |
| `participation` | `participation_threshold` (constant) | `--participation-threshold 0.1` |

`membership[i,m] = P[i,m] ≥ t_m`, then the size clamps: `--min-members 3` tops a too-small
motif up and `--max-members 0` (uncapped) truncates a too-large one, both on descending
participation. With the defaults the clamps are no-ops (member counts run 11…60), which is
what the §14 numbers require.

`extract_motif(...)` builds the signed, ordered members:

* **`rms` sign selection** — `polarity = sign(L)` when `|L| ≥ |R|`, else `sign(R)`; falls back
  to the other axis when the selected component is exactly `0`, and to `+1` when both are;
* `signed_participation = polarity · participation`;
* member order is descending `participation`, ties broken by the neuron index — a pure
  function of the arrays;
* `sender_members` = members with `dominant_axis == "left"`, `receiver_members` = `"right"`
  (disjoint, and their union is every member);
* `n_positive` / `n_negative` follow `polarity` (so `n_positive + n_negative == n_members`),
  and `n_mixed` is an additional flag counting the members whose two axes disagree
  (`sign(L) ≠ sign(R)`), hence `0 <= n_mixed <= n_members`;
* `strength_l1 = Σ participation`, `strength_energy = Σ participation²`,
  `share = strength_energy / Σ_m strength_energy`,
  `polarity_balance = (n_positive − n_negative) / n_members`.


## 7. `motifs.json` — the frozen schema (declared in full in 04A)

`src/clustering/functional_motifs.py` declares **every** constant key set in 04A and
`validate_motif_payload_schema` validates all of them — including link/family/group/pathway
entry shapes — so later sub-phases never touch these constants or the validator.

```
TOP_LEVEL_KEYS   = {provenance, config, neuron_order, motifs, links, families, groups, pathway, metadata}
PROVENANCE_KEYS  = {source_artifact, source_artifact_sha256, source_file, source_file_sha256,
                    parsed_created_utc, z_matrix_config_hash, z_matrix_config,
                    spectral_config_hash, spectral_config, phase, stage}
MEMBER_KEYS      = {index, neuron_id, participation, signed_participation, polarity,
                    left, right, dominant_axis, region, rank}
MOTIF_KEYS       = {label, mode, mode_index, value, abs_value, explained_variance_ratio,
                    cumulative_variance_ratio, threshold, threshold_method, n_members,
                    members, sender_members, receiver_members, strength_l1, strength_energy,
                    share, polarity_balance, n_positive, n_negative, n_mixed,
                    region_composition, dominant_region}
LINK_KEYS        = {source_mode, target_mode, source_label, target_label, shared_members,
                    n_shared, jaccard, polarity_agreement, classification}
FAMILY_KEYS      = {family_id, label, modes, n_modes, size, n_members, members, polarity, occurrences}
OCCURRENCE_KEYS  = {mode, participation, signed_participation}
GROUP_KEYS       = {group_id, label, members, n_members, size, dominant_mode,
                    region_composition, cent_mean, coherence, is_background, is_singleton}
PATHWAY_KEYS     = {nodes, edges, n_nodes, n_edges, weight, source, edge_threshold, top_edges,
                    intra, abs_max, n_positive, n_negative, n_intra, n_cross,
                    weight_concentration, weights}
NODE_KEYS        = {node_id, group_id, label, size, members, cent_mean, region_composition,
                    dominant_mode, top_modes}
EDGE_KEYS        = {source, target, weight, abs_weight, n_modes, modes, is_intra, polarity,
                    top_modes}
TOP_MODE_KEYS    = {mode, rank, share, signed_weight, abs_weight}
```

**Staged-field policy** (§B.2 of the master decomposition): fields owned by a later sub-phase
are emitted as their **neutral value** and later *replaced*, never added or retyped.

| Field | 04A state |
|---|---|
| `motifs` | fully populated (`M01…M21`) |
| `links` | `[]` |
| `families` | **k singletons** (`F01…F21`, one mode each) so `family_of_mode` is always defined |
| `groups` | `[]` |
| `pathway` | `{nodes: [], edges: [], weights: [], n_nodes: null, n_edges: null, …}` |
| `metadata` motif block | populated |
| `metadata` link/family, group, pathway blocks | `null` |

Schema tests therefore assert **key presence plus a type set that includes the neutral
value** (`isinstance(metadata["n_groups"], (int, type(None)))`), so they pass unchanged at
every stage. `config` and `hash_fields` are complete in 04A, so `config_hash` is identical at
every stage and a non-default config gets its filename segment from 04A onward.


## 8. `motifs.npz` — the derived array cache

Contents are a pure function of the canonical JSON bytes and keyed by that JSON's SHA-256.
16 arrays, fixed order and little-endian dtypes (`<f8` / `<i8` / `<U*`):

| Key | Shape/dtype | Meaning |
|---|---|---|
| `stage` | `(1,) <U8` | `"04A"` (04B-04D overwrite with their own stage) |
| `participation` | `(N, k) <f8` | the participation matrix |
| `membership` | `(N, k) <i8` | 0/1 membership |
| `loadings_left` | `(N, k) <f8` | Phase 03 sender basis |
| `loadings_right` | `(N, k) <f8` | Phase 03 receiver basis |
| `values` | `(N,) <f8` | the full ranked spectrum |
| `retained_values` | `(k,) <f8` | the retained mode values |
| `abs_retained_values` | `(k,) <f8` | their magnitudes |
| `explained_variance_ratio` | `(N,) <f8` | ranked EVR |
| `thresholds` | `(k,) <f8` | effective per-mode threshold |
| `motif_strength_l1` | `(k,) <f8` | `sum participation` per motif |
| `motif_strength_energy` | `(k,) <f8` | `sum participation^2` per motif |
| `family_of_mode` | `(k,) <i8` | family id per mode - identity `arange(k)` in 04A |
| `neuron_order` | `(N,) <U*` | row index order |
| `source_json_sha256` | `(1,) <U64` | digest of the JSON this cache derives from |
| `numpy_version` | `(1,) <U64` | the numpy that wrote it |

Keys are **strictly additive** across 04B-04D (a module-level verification table is extended,
never edited). Byte-reproducible (numpy pins the zip `date_time` to 1980-01-01), atomic
(in-memory bytes -> temp file -> `fsync` -> `os.replace`), always read with
`allow_pickle=False`. A stale or tampered cache is ignored with a warning and the JSON wins;
reading a tampered `.npz` directly raises `MotifValidationError`. Writes are ordered JSON
first, so a failed cache write can never lose the artifact of record.

## 9. `motifs.data.json` (`--save-data`)

Keys (exact): `provenance`, `config`, `stage`, `neuron_order`, `neurons`, `motifs`, `links`,
`families`, `groups`, `pathway`, `metadata`, `generator`, `created_utc`.

`neurons[i]` (`NEURON_PARTICIPATION_KEYS`) carries `index`, `neuron_id`, `region`, `cent`,
`out_degree`, `in_degree`, `recurrence`, `max_participation`, `mean_participation`,
`dominant_mode` and the full length-`k` `participation` row. `cent`/degrees come from the
sibling `parsed_graph.json` and are `null` when it is absent - the canonical `motifs.json`
stays self-sufficient.

## 10. Regions and anatomy

`derive_region(neuron_id)` repeatedly strips trailing tokens matching `_[LR]`, `_C<digits>`
and `_<digits>`, returning the id unchanged when nothing matches. On the reference data this
yields **31** derived regions (a single-pass strip gives 74 and `split("_")[0]` gives 29 -
both wrong).

`load_anatomy(path)` accepts a tolerant JSON: `{neuron_id: region}`, a `neurons` record list,
or a `parsed_graph.json`-shaped file. `--anatomy` overrides the heuristic; `anatomy_missing`
is informational, `anatomy_mismatch` (conflict with the derived region) is a warning.

## 11. Config and `config_hash`

`MOTIF_CONFIG_FIELDS` is the **26-field** hashed recipe spanning the whole of Phase 04, so
`config_hash` is identical at every stage:

| Owner | Fields | n |
|---|---|---|
| 04A | `participation`, `threshold_method`, `relative_threshold`, `absolute_threshold`, `quantile`, `participation_threshold`, `min_members`, `max_members` | 8 |
| 04B | `family_jaccard`, `family_polarity`, `link_min_jaccard` | 3 |
| 04C | `grouping`, `polarity_split`, `polarity_min_members`, `max_group_size`, `n_groups`, `linkage`, `affinity`, `merge_threshold` | 8 |
| 04D | `pathway_source`, `pathway_weight`, `pathway_edge_threshold`, `pathway_top_edges`, `intra`, `max_diagram_nodes`, `diagram_layout` | 7 |

`CONFIG_KEYS = MOTIF_CONFIG_FIELDS | {k_resolved, n_motifs, config_hash}`. `is_default()`
compares only the 8 04A-selectable fields, so a default run still writes the canonical
`motifs.json`. Presentation/IO flags (`--plot`, `--no-popup`, `--cmap`, `--plot-pathways`,
`--save-data`, `--stats`, `--config-hash`, `--no-sidecar`, `--dry-run`) are deliberately
**excluded** from the hash, following the Phase 02/03 precedent.

## 12. Module layout and data structures

```
src/clustering/__init__.py              # lazy __getattr__ re-exports (Phase 01-03 pattern)
src/clustering/functional_motifs.py     # 04A scope
tests/test_functional_motifs.py         # 62 fast + 2 slow
tests/test_motif_schema.py              # 42 fast
```

Dataclasses: `MotifConfig` (the 26 hashed fields + `k_resolved`/`n_motifs`/`stage`,
with `to_dict`/`from_dict`/`hash_fields`/`config_hash`/`is_default`), `MotifMember`,
`Motif`, `MotifLink`, `MotifFamily`, `NeuronGroup`, `PathwayNode`, `PathwayEdge`,
`PathwayDiagram` (with a `neutral()` classmethod), `MotifAnalysis`, and
`MotifValidationError(ValueError)` (carries `.report`).

`MotifAnalysis.metadata` holds the JSON-serializable summaries (the same dict drives `--stats`
and `--save-data`); `loaded_from` records where an object was read from and is never
serialized, so `to_dict()` round-trips byte-for-byte. Phase 01's `ValidationReport` /
`ValidationIssue` collection and the `...Error(ValueError)`-with-`.report` convention are
reused verbatim. Arrays (`participation`, `membership`, `loadings_left/right`, `values`,
`explained_variance_ratio`, `thresholds`, `motif_strength_l1/energy`) are kept on the object
for the sidecar and are never serialized into the JSON.

## 13. Algorithm

1. **Resolve** inputs (`-i`, `--glob`, `--include-variants`) into a deterministic list of
   Phase 03 JSON artifacts; directories are searched recursively and only `eigen[.variant].json`
   is accepted.
2. **Load** the spectrum (`load_spectrum`) and build `MotifConfig` from the CLI.
3. **Validate** the input with Phase 03's `validate_spectral_payload_schema`, then the config,
   the neuron order and the participation matrix.
4. **Participate**: `compute_participation(L, R, variant)`.
5. **Threshold**: `motif_thresholds(...)`.
6. **Extract**: membership + clamps, then `extract_motifs(...)`.
7. **Families**: `initial_families(motifs)` -> k singletons.
8. **Statistic**: `motif_statistics(...)` -> `metadata`.
9. **Validate the produced payload** (`validate_motif_payload_schema` + finiteness checks).
10. **Write** the JSON, then the requested derived artifacts (sidecar, save-data);
    `--dry-run` writes nothing.
11. **Report**: optional `--stats` block, then the completion box (always, on success).

## 14. Reference numbers (verified on the real artifact; 04A owns these)

| Quantity | Value |
|---|---|
| `N` / `k` / `n_motifs` | 113 / 21 / 21 |
| member counts per mode (ranks 1-21) | 33, 23, 18, 30, 16, 12, 19, 22, 11, 43, 41, 19, 60, 16, 17, 18, 32, 39, 55, 39, 60 |
| `n_members_total` / min / median / max / mean | 623 / 11 / 23 / 60 / 29.666667 |
| `n_unique_members` / `n_neurons_in_no_motif` | 107 / 6 |
| `membership_density` | 623 / (113*21) = **0.26253687315634217** |
| recurrence >=1 / >=2 / >=5 / max | 107 / 99 / 70 / 13 |
| top recurrent (descending, ties by neuron index) | FB4Z_R_1 (13), hDeltaA_12_C10_1 (13), FB4Z_R_2 (12), hDeltaA_08_C7_1 (12), hDeltaA_10_C8_1 (12), hDeltaA_11_C9_1 (12) |
| background neuron | `DNa03_R_1` (zero participation => `n_isolated_neurons = 1`) |
| neurons in no motif | DNa03_R_1, MBON09(y3B'1)(AVM17)_L_1, MBON09(y3B'1)(AVM17)_L_2, SMP192(PDL05)_L_1, SMP192_R_1, hDeltaI_12_C10_2 |
| derived regions from ids | **31** |
| `n_families` (04A state) | k singletons - **asserted as the partition invariant, not as a fixed count** |

Corrections applied to the first draft of this plan (all re-verified):

1. `membership_density` is `0.26253687315634217` - the draft's `0.262482` was a rounding slip.
2. Four neurons tie at recurrence 12; the draft listed only three of them and omitted
   `hDeltaA_08_C7_1`.
3. `motifs.npz` carries **16** arrays (the master decomposition's "15 keys" row was off by one).

## 15. CLI reference

```
python -m src.clustering.functional_motifs \\
  -i data/processed/<stem>/eigen.json -o data/processed \\
  [--glob 'eigen.json'] [--include-variants] [--anatomy PATH] \\
  [--participation {left,right,max,rms}] \\
  [--threshold-method {relative,absolute,quantile,participation}] \\
  [--relative-threshold 0.25] [--absolute-threshold 0.05] [--quantile 0.8] \\
  [--participation-threshold 0.1] [--min-members 3] [--max-members 0] \\
  [--config-hash] [--no-sidecar] [--save-data] [--stats] \\
  [--strict] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

Exit codes `0` success, `1` any input failed validation/writing, `2` no input matched.
Presentation/grouping/pathway flags are deliberately **absent** so they cannot silently
no-op; they arrive with their owner.

## 16. Terminal output

`--stats` (terminal only) prints the participation mode, the threshold method, `N x k`, the
motif count, member totals/unique/none, membership density, the recurrence histogram, the top
recurrent neurons, the region count and explicit `not computed yet (04B/04C/04D)` lines -
never zeros. The post-run box lists the participation mode, the threshold method, `n_motifs`,
member totals, the JSON/NPZ/save-data lines and the stage, and reflects the real outcome
(`skipped (--no-sidecar)`, `skipped (--dry-run)`).

## 17. Validation rules

| Check | Severity |
|---|---|
| Input conforms to the Phase 03 schema (reuses `validate_spectral_payload_schema`) | Error |
| Config enums/ranges for the 04A fields | Error |
| `neuron_order` unique, non-empty; `values`/`loadings` lengths consistent | Error |
| Participation finite and non-negative | Error |
| Emitted payload conforms to the Phase 04 schema | Error |
| Sidecar/save-data write failure (never a partial file) | Error |
| `empty_motif` (a motif with no member) | Warning (`--strict` => error) |
| `small_graph` (`N < 2`) | Warning (`--strict` => error) |
| `trivial_motifs` (every participation value zero) | Warning (`--strict` => error) |
| `anatomy_missing` / `anatomy_mismatch` | Info / Warning |
| Degenerate (all-zero) mode column | Informational log line |

## 18. Invariants (asserted in tests)

1. `participation.shape == (N, k)`, finite, non-negative; `membership` in `{0,1}`.
2. `sum_m n_members == n_members_total`; every motif's `n_members == len(members)`.
3. `n_positive + n_negative == n_members`, `0 <= n_mixed <= n_members`;
   `sender | receiver = members`, disjoint.
4. Member order is monotone in `participation` (ties by neuron index).
5. `share` sums to 1; `strength_l1`/`strength_energy` match the members' values.
6. `families` partition all `k` modes; `family_of_mode` is the identity.
7. `trivial_motifs`/`empty_motif` behave exactly as documented.
8. `motifs.json` and `motifs.npz` are byte-identical across runs under a fixed
   `SOURCE_DATE_EPOCH`; `load_motifs(path).to_dict()` reproduces the artifact byte-for-byte.
9. The npz digest matches the JSON bytes; `--no-sidecar` leaves one artifact and
   `load_motifs()` still works; a tampered cache never corrupts a load.
10. The Phase 03 object and its payload are **never mutated**; **no `*.csv`** is produced.

## 19. Testing strategy

Framework: `pytest` 9.1.1 with the existing `pytest.ini` (no new markers, no plugin
assumptions). Baseline **397** (Phase 01-03) -> **503**. 04A adds **104 fast + 2 slow**
(`tests/test_functional_motifs.py` 62 fast + 2 slow, `tests/test_motif_schema.py` 42 fast).

* *Participation*: all four variants on synthetic `(8, 3)` arrays; `rms` sign selection;
  zero-row detection; finiteness/dtype.
* *Thresholds*: all four methods against hand-computed values on a synthetic `8x3` spectrum;
  `min`/`max-members` clamps; the default reproduces the exact reference counts.
* *Members*: ordering, strength/share/polarity-balance identities, sender/receiver partition.
* *Regions*: the strip heuristic, `--anatomy` override, `parsed_graph.json` enrichment, the
  31-region reference count.
* *Families*: `initial_families` unit test + the partition invariant.
* *Pipeline*: provenance/config/`config_hash`; JSON+NPZ determinism under `SOURCE_DATE_EPOCH`;
  sidecar round-trip, stale and tampered cache; `--no-sidecar`; `--dry-run`; exit codes
  `0/1/2`; `eigen.data.json` rejection; no-CSV; `python -m` smoke test.
* *Schema*: the complete frozen schema plus every negative case at every level.
* *Slow*: the section-14 block on the real `eigen.json`, plus the `eigen.f8652585.json` variant.

## 20. Acceptance criteria

1. A default run on the reference artifact reproduces every section-14 number with zero errors.
2. Artifacts: `motifs.json` always; `motifs.npz` unless `--no-sidecar`; `motifs.data.json`
   with `--save-data` - **no CSV**.
3. The frozen schema validator is green on the real payload; every negative case is caught.
4. `load_motifs()` and `load_sidecar_arrays()` round-trip; the cache is digest-keyed.
5. Both artifacts are byte-identical under a fixed `SOURCE_DATE_EPOCH`.
6. All 397 Phase 01-03 tests still pass unchanged; no Phase 01-03 file/artifact is touched.

## 21. Non-goals

Link classification and family merging (04B); functional groups (04C); pathway graph, plots,
figures and Phase 05 handoff (04D); interactive windows; any sklearn / networkx / matplotlib
import; CSV output; multi-file union.

## 22. Deliverables

* `execution-plans/04A_motifs.md` (this document)
* `src/clustering/__init__.py`, `src/clustering/functional_motifs.py`
* `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`
* `README.md` Phase 04 section + Status row
* Artifacts: `data/processed/<stem>/motifs.json`, `motifs.npz` (+ `motifs.data.json`)

## 23. How 04B-04D extend this

04B populates `links` + merged `families` and appends `family_of_mode`, `link_jaccard`,
`link_polarity` to the npz; 04C populates `groups`; 04D populates `pathway` and adds the
figures. None of them touch the 04A key sets, the validator, the 26 config fields or the
`config_hash`.
