# Sub-phase 04B — Cross-mode motif tracking (links & families)

*Implementation: `src/clustering/functional_motifs.py` (with `src/clustering/__init__.py`).*
*Artifacts: the same `data/processed/<stem>/motifs.json`, `motifs.npz`, `motifs.data.json`.*
*Tests: `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`.*

> **Summary (the brief this document expands).** Same input; `links` and `families`
> populated in `motifs.json`; `family_of_mode`, `link_jaccard` `(k,k)`, `link_polarity`
> `(k,k)` added to `motifs.npz`; the link/family blocks of `metadata` and of
> `motifs.data.json` populated. New symbols `motif_links`, `classify_link`,
> `merge_families`, `family_members`, `recurrence_table`. New CLI flags
> `--family-jaccard` (0.2), `--family-polarity` (0.5), `--link-min-jaccard` (0.1).
> Reference numbers: `n_links` 187, stable/flipped/composite/weak 1/2/133/51,
> `n_families` **18**, multi-mode families `{2,15}`, `{8,13}`, `{11,21}`.

---

## 1. Goals

Decide **whether and how modes share subcircuits**: the pairwise link table and the merged
motif families. 04B answers exactly one question — *do modes share motifs?* — and consumes
only 04A's per-mode structure. It owns the `links`/`families` layer and nothing else.

Pipeline grows by one stage, still fixed and recorded in every artifact:

```
load spectrum -> participation -> thresholds -> membership -> signed members
-> links -> families -> statistics
```

**Purely additive.** No existing public function changes signature or semantics; the frozen
key sets, the schema validator, the 26 config fields and `config_hash` are untouched. No new
dependency (numpy only).

## 2. Scope boundary

| Owned by 04B | Explicitly **not** 04B |
|---|---|
| `links`, merged `families`, link metadata block, `--family-*`/`--link-*`, tracking stats/box lines | functional groups (04C) |
| `link_jaccard`, `link_polarity`, merged `family_of_mode` in `motifs.npz` | pathway graph + figures (04D) |
| `motif_links`, `classify_link`, `merge_families`, `family_members`, `recurrence_table` | any schema/config change; sklearn/networkx/matplotlib |

## 3. Inputs

`eigen[.<variant>].json` through `load_spectrum()` — exactly as 04A. 04B re-reads the same
input and rebuilds the participation/membership/motifs deterministically, then adds the
cross-mode layer. Optional `parsed_graph.json`/`--anatomy` enrichment is unchanged.

## 4. Outputs

Same artifact set, richer content:

| Artifact | 04B change |
|---|---|
| `motifs.json` | `links` populated; `families` merged; `metadata` link block filled; `provenance.stage = "04B"` |
| `motifs.npz` | 16 → **18** arrays (`link_jaccard`, `link_polarity` added; `family_of_mode` re-derived) |
| `motifs.data.json` | `links`/`families` filled automatically (the payload already forwards them) |

## 5. Links — the core definition

For every **unordered** pair of modes `(i, j)` with a non-empty member intersection:

* `shared_members` — the intersection in ascending neuron index;
* `n_shared = |shared_members|`;
* `jaccard = |∩| / |∪|`;
* `polarity_agreement = (n_same − n_diff) / n_shared` — **signed**, in `[-1, 1]`, where
  `n_same` counts shared neurons whose `polarity` matches in both motifs;
* `classification = classify_link(...)` (below).

`links` contains exactly the pairs with `n_shared > 0` (187 of `C(21,2) = 210` on the
reference data), ordered by `(source_mode, target_mode)`.

`classify_link(jaccard, polarity_agreement, *, family_jaccard, family_polarity, link_min_jaccard)`:

| Class | Condition | Meaning |
|---|---|---|
| `weak` | `jaccard < link_min_jaccard` | barely any overlap; still reported |
| `stable` | `jaccard >= family_jaccard` and `polarity_agreement >= family_polarity` | strong, sign-consistent overlap |
| `flipped` | `jaccard >= family_jaccard` and `polarity_agreement <= -family_polarity` | strong overlap with opposite polarity |
| `composite` | everything else | reportable overlap that may **not** merge |

All three thresholds are **inclusive** (`weak` uses strict `<`, so a link exactly at
`link_min_jaccard` is `composite`). The inclusive polarity band is required: the reference
pair `(2,15)` has `polarity_agreement = -0.50` exactly and must classify as `flipped`.

## 6. Families — merging

`merge_families(motifs, links)` — the thresholds are already baked into each link's
`classification`, so the merge needs nothing else:

1. **Union-find over `stable` + `flipped` links only.** `composite`/`weak` links never merge;
   including them would collapse the mode graph into a single giant component (on the
   reference data: all 21 modes in one component).
2. Components are ordered by their lowest mode; each is labelled `F01`, `F02`, ... with a
   1-based `family_id`, so `families` stays a **partition of all `k` modes**.
3. **Polarity orientation.** The lowest mode is the representative (sign `+1`); a `stable`
   edge keeps the sign, a `flipped` edge negates it (BFS). A contradiction is logged through
   `CODE_FAMILY_CONFLICT`; the first assignment wins, so the result stays deterministic.
4. `members` is the union of the component's members, ordered by `(mode asc, member rank
   asc)`, de-duplicated; `occurrences` records every `{mode, participation,
   signed_participation}` with the family sign applied (a flipped mode contributes
   sign-aligned values).
5. `polarity` is the mean of the component's **sign-normalised** `polarity_balance`. For a
   singleton this equals the 04A `motif.polarity_balance` exactly, so unmerged families are
   byte-identical to 04A.

## 7. Payload changes (schema unchanged)

| Block | 04A | 04B |
|---|---|---|
| `links` | `[]` | 187 entries, all `LINK_KEYS` |
| `families` | k singletons | merged families, all `FAMILY_KEYS` |
| `metadata.n_links` / `link_classes` | `null` | `187` / `{"stable":1,"flipped":2,"composite":133,"weak":51}` |
| `metadata.n_families` / `n_multi_mode_families` | `null` | `18` / `3` |
| `provenance.stage`, `config.stage`, npz `stage` | `"04A"` | `"04B"` |

`config_hash` is **unchanged** (the three fields were already in `MOTIF_CONFIG_FIELDS`).

## 8. `motifs.npz` additions

| Key | Shape/dtype | Meaning |
|---|---|---|
| `family_of_mode` | `(k,) <i8` | family index per mode (0-based; identity for singletons) |
| `link_jaccard` | `(k,k) <f8` | symmetric Jaccard matrix, unit diagonal, `0` where no shared member |
| `link_polarity` | `(k,k) <f8` | symmetric polarity-agreement matrix, unit diagonal |

All three are JSON-derivable, so `_expected_sidecar_arrays` (the cache verification table)
gains the two matrices and derives `family_of_mode` from `analysis.families`. That derivation
is the identity for a 04A payload, so a 04A-produced cache still verifies; a cache written
before this sub-phase lacks the two new keys and is treated as stale (warned and ignored, the
JSON wins), exactly as designed.


## 9. `metadata`, `motifs.data.json` and `recurrence_table`

The 04B metadata block is produced by the private `_tracking_statistics(links, families)` and
merged into the 04A metadata (the existing `motif_statistics` signature is untouched, so the
direct-call unit test keeps working): `n_links`, `link_classes` (all four keys, zero-filled),
`n_families`, `n_multi_mode_families`. `motifs.data.json` already forwards `analysis.links`
and `analysis.families`, so it becomes richer with no code change.

`recurrence_table(motifs, families=None)` returns
`{neuron_id: {n_modes, modes, n_families, families, max_participation,
max_signed_participation}}` (sorted lists). It is **not** serialized into the frozen
`NEURON_PARTICIPATION_KEYS` table; it is exposed for 04C and stored in
`analysis.diagnostics["recurrence_table"]` (diagnostics are never serialized).

## 10. New symbols

Public: `classify_link`, `motif_links`, `merge_families`, `family_members`,
`recurrence_table`, `LINK_CLASSES`, `CODE_NO_LINKS`, `CODE_FAMILY_CONFLICT`.
Private: `_member_index`, `_family_of_mode`, `_link_matrices`, `_tracking_statistics`.
`STAGE` becomes `"04B"`.

```python
def classify_link(jaccard, polarity_agreement, *, family_jaccard=..., family_polarity=...,
                  link_min_jaccard=...) -> str
def motif_links(motifs, config=None, *, family_jaccard=None, family_polarity=None,
                link_min_jaccard=None) -> list[MotifLink]
def merge_families(motifs, links) -> list[MotifFamily]
def family_members(modes, by_mode, signs=None) -> tuple[tuple[str, ...], tuple[dict, ...]]
def recurrence_table(motifs, families=None) -> dict[str, dict]
```

## 11. Algorithm

After 04A's `extract_motifs`:

1. `links = motif_links(motifs, config)`; log `no_links` (info) when empty.
2. `merged = merge_families(motifs, links)`; `families = [f.to_dict() for f in merged]`.
3. Build the 04A metadata, then `metadata.update(_tracking_statistics(links, merged))`.
4. Inject `family_of_mode`, `link_jaccard`, `link_polarity` into `analysis.arrays` (the same
   pattern used for `source_json_sha256`/`numpy_version`), so `_build_arrays` is untouched.
5. Validate the payload with the frozen validator, then write JSON → NPZ → save-data.

## 12. Config and CLI

The three fields already exist in `MOTIF_CONFIG_FIELDS` (hashed) and in `MotifConfig`, so
`config_hash` is unchanged. A new `cross-mode tracking` argument group adds
`--family-jaccard` (0.2), `--family-polarity` (0.5), `--link-min-jaccard` (0.1); they are
passed through `_config_from_args`.

> **Known limitation (inherited from 04A).** `MotifConfig.is_default()` compares only the
> eight 04A-selectable fields (`DEFAULT_DECISION_FIELDS`), and 04B deliberately does not
> change that. A run with non-default tracking thresholds therefore keeps the canonical
> `motifs.json` name unless `--config-hash` is given; use `--config-hash` for a safe
> non-default tracking run. The `config_hash` itself already changes, and is recorded.

## 13. Validation rules

| Check | Severity |
|---|---|
| `family_jaccard` / `family_polarity` / `link_min_jaccard` in `[0, 1]` | Error |
| `link_min_jaccard <= family_jaccard` | Error |
| `no_links` (no pair shares a member) | Info (log) |
| `family_conflict` (contradictory polarity inside a component) | Warning (log) |
| Emitted payload conforms to the frozen schema | Error |

## 14. Invariants (asserted in tests)

1. `families` partitions all `k` modes (disjoint, sorted, complete) for any `config`;
   `family_of_mode[m]` is the index of the family containing mode `m`.
2. `n_links == len(links) == Σ link_classes`; `n_families == len(families)`;
   `n_multi_mode_families == #{family : n_modes > 1}`.
3. Only `stable`/`flipped` links merge; `composite`/`weak` links never do.
4. `link_jaccard`/`link_polarity` are symmetric with a unit diagonal; `0` off-diagonal where
   no member is shared.
5. A singleton family reproduces the 04A `polarity_balance`, `members` and `occurrences`.
6. A flipped family makes shared members' `signed_participation` sign-consistent.
7. `motifs.json`/`motifs.npz` are byte-identical across runs under a fixed `SOURCE_DATE_EPOCH`.
8. The 04A object and its payload are never mutated; no `*.csv` is produced.

## 15. Reference numbers (verified on the real artifact; 04B owns these)

| Quantity | Value |
|---|---|
| `n_links` | **187** (all pairs with `shared > 0`; `C(21,2) = 210`) |
| stable / flipped / composite / weak | **1 / 2 / 133 / 51** |
| `n_families` | **18** |
| `n_multi_mode_families` | **3** |
| multi-mode families | `F02 = [2,15]` (flipped, 32 members), `F08 = [8,13]` (flipped, 66), `F11 = [11,21]` (stable, 76) |
| `family_of_mode` | `[0,1,2,3,4,5,6,7,8,9,10,11,7,12,1,13,14,15,16,17,10]` |
| `link_jaccard` / `link_polarity` | `(21,21)`, symmetric, unit diagonal |
| documented finding | cross-mode recurrence is weak by construction: the Phase 03 bases are orthonormal, so modes share only a minority of members (mean pairwise Jaccard ≈ 0.197, max 0.619) |


## 16. Testing strategy

Framework: `pytest` with the existing `pytest.ini` (no new markers). 04B adds **39 fast + 1
slow** across the two modules.

* `tests/test_functional_motifs.py` — `classify_link` at every exact boundary
  (`link_min_jaccard`, `family_jaccard`, `±family_polarity`) and with custom thresholds;
  `motif_links` coverage/ordering and the disjoint (`no_links`) case; `merge_families` unit
  tests (only merging links, labels/ids, empty-link singletons); the **giant-component
  regression** (`composite` links must not create a family); flipped-family sign alignment;
  `family_members` union/occurrences; `recurrence_table`; the link matrices' symmetry and
  unit diagonal; tracking metadata + save-data; config rejection; the partition invariant
  for several configs; CLI flags reach the config; artifact determinism.
* `tests/test_motif_schema.py` — link/family entry conformance and value ranges; the family
  partition; metadata link-block types and cross-checks; the three npz arrays and the
  `load_motifs` round-trip.
* **Slow** — `test_reference_tracking_numbers`: 187 / 1-2-133-51 / 18 / `{2,15},{8,13},{11,21}`
  / the exact `family_of_mode` / the matrix shapes, on the real `eigen.json`.

## 17. Required adjustments to 04A test assertions

04B legitimately populates values the 04A tests pinned at their **pre-04B neutral state** (the
"legitimate exception" anticipated by the master decomposition's §B.3). These are assertion
updates only — no 04A behaviour changes:

| File | Assertion | New form |
|---|---|---|
| `test_motif_schema.py` | `payload["links"] == []`; link metadata keys `is None` | links non-empty; link metadata typed and cross-checked |
| `test_functional_motifs.py` | `len(arrays) == 16` | `== 18` |
| `test_functional_motifs.py` | `family_of_mode == [0,1,2]`, `families[0].n_modes == 1`, `set(family_of_mode) == set(range(k))` | derived from `families` (the synthetic 3-mode fixture merges into one family under the inclusive boundary) |
| `test_functional_motifs.py` | `"Phase 04A complete"` / `"Phase 04A -- motif statistics"` / `"not computed yet (04B)"` | `04B` titles; the 04B line is now real, `04C`/`04D` stay "not computed yet" |
| `test_functional_motifs.py` (slow) | `metadata["n_links"] is None`, `len(families) == 21` | `== 187`, `== 18` (the partition invariant is unchanged) |

Every other 04A test — participation, thresholds, members, regions, provenance, determinism,
exit codes, `--dry-run`, sidecar staleness, `initial_families`, the direct
`motif_statistics(...)` call — is unchanged and green.

## 18. Acceptance criteria

1. A default run on the reference artifact reproduces every §15 number with zero errors.
2. `n_families == 18` while the partition invariant holds; `family_of_mode` is consistent
   with `families`.
3. The frozen schema validator is green on the real payload.
4. `link_jaccard`/`link_polarity` are symmetric with a unit diagonal; `load_motifs()` and
   `load_sidecar_arrays()` round-trip.
5. Artifacts are byte-identical under a fixed `SOURCE_DATE_EPOCH`.
6. The full suite is green; `git diff --stat` touches only the 04B files.

## 19. Non-goals

Functional neuron groups (04C); pathway graph, figures and Phase 05 handoff (04D); any new
schema key or config field; sklearn/networkx/matplotlib; CSV output.

## 20. Deliverables

* `execution-plans/04B_tracking.md` (this document)
* `src/clustering/functional_motifs.py`, `src/clustering/__init__.py`
* `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`
* `README.md` — Phase 04B tracking subsection + Status row
* Artifacts: the regenerated `motifs.json` / `motifs.npz` (+ `motifs.data.json`)

## 21. How 04C–04D extend this

04C populates `groups` from the motif membership (and, opt-in, the loading vectors) and adds
`group_labels`, `group_centroids`, `group_sizes` to the npz; 04D populates `pathway` and adds
the figures. Neither touches the 04A/04B key sets, the validator, the 26 config fields or the
`config_hash`. `recurrence_table` and `family_of_mode` are the natural inputs for 04C's
membership-based grouping.

