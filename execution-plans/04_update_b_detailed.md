# Phase 04 Update B — unified pathway scoring (S-score): detailed design

*Implementation: `src/clustering/motif_analysis.py` (new module, read-only) with
`src/clustering/__init__.py` lazy re-exports.*
*Artifacts: `motif_analysis_summary.json` (always) and `motif_analysis_summary.csv`
(`--csv` only), both written into the existing run directory.*
*Tests: `tests/test_motif_analysis.py` (61 fast + 3 slow).*

> This document is the detailed design for the spec in `execution-plans/04_update_a.md`
> (titled "Phase 04 Update B").  The spec file is left intact, mirroring the
> `04C_grouping.md` / `04C_grouping_detailed.md` split.  It records the locked decisions, the
> exact component semantics, and the three places where the spec as written could not be
> implemented literally (each verified against the real artifacts).

---

## 1. Locked decisions

| # | Decision |
|---|---|
| **B-1** | `motif_analysis_summary.csv` is **opt-in only** via `--csv` (default off). It is the single, documented exception to the project's "no CSV anywhere" invariant; the five existing no-CSV tests stay green because they never pass `--csv`. |
| **B-2** | **Phase 02 `z_matrix[.<variant>].json` is in scope**: the scorer rebuilds `W_matrix` from it (`Iᵀ·Z·I`, the same reduction 04D uses) and inverts 04D's blend algebraically to recover `W_mode`. Phase 03 `eigen.json` is **not** read. |
| **B-3** | `matrix-only` (and `hybrid` with `α = 0`): `D = null`, the term is dropped from `S`, and the row records `d_unavailable_reason` (`"matrix-only: no mode contribution stored"`). |

## 2. Scope and contract

* **New file only.** Zero edits to `src/clustering/functional_motifs.py`; the frozen key sets,
  `validate_motif_payload_schema`, `MotifConfig`, `MOTIF_CONFIG_FIELDS` and `config_hash` are
  untouched.  `git diff --stat` shows the new module, the new test module, the lazy re-exports
  in `src/clustering/__init__.py`, the README and this document.
* **Reader, never a writer of Phase 04 artifacts.** `motifs.json`, `motifs.npz`,
  `motifs.data.json` and `motifs.pathway.graphml` are never rewritten (asserted by a test that
  byte-compares every `motifs.*` file before/after scoring).
* **No new dependency** (stdlib `csv` + numpy, already pinned).

```
src/clustering/motif_analysis.py     # discovery, S-score, ranking, summary, CSV, CLI
tests/test_motif_analysis.py         # 61 fast + 3 slow
```

## 3. The S-score

```
S = w_B·B + w_H·H + w_D·D − w_C·C          # defaults: every weight = 1
```

| Component | Definition | Range / notes |
|---|---|---|
| `B` | `1 − \|n_pos − n_neg\| / n_edges` | `[0, 1]`; the denominator is `n_edges` (so zero-polarity edges act as a mild bonus, exactly as specified) |
| `H` | `−Σ_s p_s · ln p_s`, `s ∈ {−1, 0, +1}`, `p_s = count_s / n_edges` | `[0, ln 3 ≈ 1.0986]`; `p_s = 0` terms are **skipped**, so `H` is defined when `p_zero = 0` |
| `D` | `#(sign(W_mode) ≠ sign(W_matrix)) / n_edges`, over the **kept** edges | `[0, 1]`; `null` when not recoverable |
| `C` | `pathway.weight_concentration` verbatim | 04D's `abs_max / Σ\|w\|` over the full `G×G` matrix |

An undefined component is reported as `null` and its term is **dropped** from `S` (never
coerced to `0`), with `d_unavailable_reason` recording why.  An edge-free pathway (e.g.
`--grouping none`) receives **no score** (`S = null`), so a degenerate run never masquerades
as `S = 0`.  Weights must be finite and `≥ 0` (`CODE_SCORE_CONFIG` otherwise).
`-0.0` is canonicalized to `0.0` because `H = 0` and `S = −C` both produce a `-0.0` naturally.

### 3.1 Recovering `W_mode` (the crux)

`motifs.json` stores only the final `weight` (`W_viz`) per edge, never the mode/matrix split.
04D's blend is a fixed closed form, so the split is invertible given `W_matrix`:

| `pathway_weight_rule` | `pathway.source` | `W_mode` | `D` |
|---|---|---|---|
| `signed`/`abs`/`positive`/`negative` | `mode` | `W` | `null` (`mode-only: no matrix contribution`) |
| `signed`/`abs`/`positive`/`negative` | `both` | `W − W_matrix` | computed |
| `mode-only` | `mode` | `W` | `null` |
| `matrix-only` | `matrix` | not recoverable | `null` (B-3) |
| `hybrid` (`α > 0`) | `hybrid` | `(W − (1−α)·W_matrix) / α` | computed |
| `hybrid` (`α = 0`) | `matrix` | not recoverable | `null` (B-3) |

This is sound because the `positive`/`negative`/`abs` rules store the **untransformed**
`W = W_mode + W_matrix` as `W_viz` (only the filter weight is transformed), and `matrix-only`
stores `W = W_matrix` while emitting `modes: []` / `top_modes: []` — which is exactly why its
mode part is unrecoverable.

`W_matrix[A,B] = Σ_{i∈A} Σ_{j∈B} Z[i,j]` is rebuilt with the same `Iᵀ·Z·I` indicator reduction
04D's `_group_pair_sum` uses, so the group sums agree with the pathway that was built.
`α < 0.1` adds an informational note about the ill-conditioned inversion.

## 4. Spec deviations (each one verified numerically)

| # | Spec as written | Verified reality | Resolution |
|---|---|---|---|
| 1 | `motif_analysis_summary.csv` is listed among the outputs | The project has an absolute, test-enforced no-CSV invariant (5 tests) | **B-1**: opt-in `--csv`, documented as the single exception |
| 2 | "The script does not inspect Phase 02 or Phase 03 artifacts" **and** `D` is defined for `both`/`hybrid`/`matrix-only` | `D` is not derivable from `motifs.json` alone (`z_contribution` lives only in the GraphML, and the per-mode breakdown of a `matrix-only` run is empty) | **B-2**: read the sibling Phase 02 `z_matrix[.<variant>].json` (explicitly authorised); never read Phase 03 |
| 3 | `D` is defined for `matrix-only` | a `matrix-only` artifact stores no mode contribution at all (`modes = []`, `top_modes = []`), so `D` has no operands | **B-3**: `D = null` + `d_unavailable_reason` |
| 4 | §Input Directory lists `motifs.data.json` / `motifs.<hash>.data.json` among the inputs | `motifs.data.json` carries a **byte-identical** `pathway` **and** `config` block (verified), so scoring it too would double-count the same run | `.data.json` files are de-duplicated and skipped unless `--include-data` |
| 5 | (implied) a missing Phase 02 artifact is fatal | the `outdir/` scratch runs keep only `motifs.*`; a hard error there would make whole-directory scoring unusable | missing sibling matrix ⇒ **warning** (`CODE_SCORE_MATRIX`) with `D = null`; a matrix that **exists but mismatches** the artifact's neuron order ⇒ **error** |

## 5. Payload, CLI and artifacts

`PathwayScore` record fields: `rank`, `artifact`, `artifact_path`, `artifact_kind`,
`artifact_sha256`, `config_hash`, `stage`, `pathway_source`, `pathway_weight_rule`,
`pathway_polarity_rule`, `hybrid_alpha`, `n_nodes`, `n_edges`, `n_positive`, `n_negative`,
`n_zero`, `n_intra`, `n_cross`, `abs_max`, `weight_concentration`, `balance`, `entropy`,
`disagreement`, `concentration`, `s_score`, `weights_used`, `d_unavailable_reason`, `issues`.

`motif_analysis_summary.json` keys (exact): `provenance` (`phase`, `stage`, `generator`,
`scorer_version`, `source_directory`, `n_artifacts`, `n_scored`, `created_utc`),
`scorer_config` (the four weights, `definition`, `score_hash`), `summary` (`n_artifacts`,
`n_scored`, `n_skipped`, `s_max`, `s_min`, `s_mean`, `best_artifact`, `worst_artifact`),
`rankings` (the `PathwayScore` records sorted by `S` descending, ties on artifact name, ranks
assigned; unscored rows last with `rank = null`), `metadata` (`n_errors`, `n_warnings`,
`issues[]`, `definition`).  Written with `dumps_json`/`write_json_atomic`; `created_utc`
honours `SOURCE_DATE_EPOCH`; the CSV uses a fixed `CSV_COLUMNS` header, `lineterminator="\n"`
and an atomic text write.

| Option | Default | Behaviour |
|---|---|---|
| `-i/--input` (repeatable) | required | a run directory (or a single artifact); directories are scanned, then searched recursively when they hold no artifact |
| `--include-data` | off | also score `motifs.*.data.json` |
| `--w-balance`, `--w-entropy`, `--w-disagreement`, `--w-concentration` | `1` | S-score weights (finite, `≥ 0`) |
| `--csv` | off | write `motif_analysis_summary.csv` |
| `--stats` | off | print the S-score statistics block |
| `--strict` | off | escalate warnings (e.g. a missing sibling matrix) to errors |
| `--dry-run` | off | score and print, write nothing |
| `--log-level` | `INFO` | verbosity |

Exit codes: `0` success, `1` a failed artifact / strict escalation / write failure / invalid
weights, `2` no motif artifact matched.  On failure nothing is written (the summary is written
only after the error check), which keeps `--strict` from leaving a half-trusted artifact.

## 6. Validation rules and invariants

| Check | Severity / code |
|---|---|
| artifact unreadable, not an object, no `pathway`/`config`, `pathway.edges` not a list | Error `CODE_SCORE_INPUT` |
| `pathway.n_edges`/`n_positive`/`n_negative` inconsistent with the edge list | Error `CODE_SCORE_INPUT` |
| `weight_concentration` non-finite | Error `CODE_SCORE_INPUT` |
| sibling matrix exists but its neuron order ≠ the artifact's | Error `CODE_SCORE_MATRIX` |
| sibling matrix missing while `D` is defined | Warning `CODE_SCORE_MATRIX`, `D = null` |
| edge-free pathway | Warning `CODE_SCORE_EMPTY`, `S = null` |
| negative / non-finite / malformed weight | Error `CODE_SCORE_CONFIG` |
| `hybrid` with `0 < α < 0.1` | Info, recorded in `issues` |
| summary / CSV write failure | Error `CODE_SCORE_WRITE` (atomic, never partial) |

Invariants (asserted): `0 ≤ B ≤ 1`, `0 ≤ H ≤ ln 3`, `0 ≤ D ≤ 1`, `0 ≤ C ≤ 1`;
`B = 0` iff a single sign; `H = 0` iff `≤ 1` polarity class; `S` is a pure function of
`(B, H, D, C, weights)`; a `null` term is dropped, not zero-filled; the same directory is
byte-identical under a fixed `SOURCE_DATE_EPOCH`; scoring never mutates any existing artifact;
`--csv` is the only way a CSV can appear.


## 7. Reference numbers (verified, asserted by the slow tests)

**Canonical run** — `data/processed/FB4Yaffect_FB45_999prePost_001_all/motifs.json`
(`source = mode`, `rule = signed`): 22 nodes / 39 edges, 0 positive / 39 negative,
`B = 0.0`, `H = 0.0`, `D = null` (`mode-only: no matrix contribution`),
`C = 0.05584213591659331`, **`S = −0.05584213591659331`**.

`S` is negative here because the reference pathway is entirely negative — the documented
Phase 02 negative-weight bias (77 % of unified weights negative).  This is the correct
consequence of the specified formula, not a defect; `S` is not normalised and must only be
compared at equal weights.

**`outdir` variants** (scored with the sibling `z_matrix.json`):

| artifact | rule / source | edges | pos/neg | `B` | `H` | `D` | `C` | `S` |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `motifs.086daabf.json` | `matrix-only` / `matrix` | 78 | 13/65 | 0.333333 | 0.450561 | `null` | 0.031577 | **0.752318** |
| `motifs.4e9b4ea8.json` | `abs` / `both` | 75 | 12/63 | 0.320000 | 0.439670 | **0.013333** | 0.033013 | **0.739990** |
| `motifs.7a609e43.json` | `signed` / `mode` | 49 | 0/49 | 0.000000 | 0.000000 | `null` | 0.041491 | **−0.041491** |
| `motifs.fee5dee8.json` | `positive` / `mode` | 16 | 16/0 | 0.000000 | 0.000000 | `null` | 0.097347 | **−0.097347** |

`0.013333333333333334 = 1/75` on the only `both` run: exactly one kept edge disagrees between
the mode and matrix contributions, which is a genuinely informative result rather than a
sentinel.  Default weight profile: `score_hash = 7835a30e`.

## 8. Testing

Framework: the existing `pytest.ini` (no new markers, no plugins).  **703 → 767** tests
(**61 fast + 3 slow** new).  Coverage: every component on hand-computed fixtures including the
`0·log 0`, edge-free and single-class boundaries; the full rule × source `W_mode` inversion
matrix including `hybrid(α = 1) ≡ mode-only` and `hybrid(α = 0) ≡ matrix-only`; `W_matrix`
against a brute-force nested sum; weight validation; discovery / de-duplication / variant
resolution (including the `variant`-vs-`config_hash` ambiguity, resolved by existence);
ranking order and tie-breaks; summary shape and byte determinism; `--csv` opt-in (no CSV
without it); CLI exit codes `0/1/2`, `--dry-run`, `--strict`, and a `python -m` subprocess
smoke test; a full-pipeline integration test that builds real `motifs.*` artifacts from the
synthetic spectrum and asserts the scorer left them byte-identical; and the §7 reference block
as slow tests.

Two design gaps were found by the tests during implementation and fixed in the module:
the variant-vs-`config_hash` ambiguity in sibling-matrix resolution (now resolved by
existence over an ordered candidate list) and the edge-free pathway scoring `S = 0.0`
(now explicitly `S = null`).

## 9. Verification protocol (run and green)

```bash
PY=/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python
$PY -m pytest -m "not slow" -q     # 736 passed, 31 deselected
$PY -m pytest -q                   # 767 passed
$PY -m src.clustering.motif_analysis -i data/processed/FB4Yaffect_FB45_999prePost_001_all --stats --dry-run
SOURCE_DATE_EPOCH=1700000000 … (twice, same directory)   # byte-identical JSON + CSV
git --no-pager diff --stat         # functional_motifs.py and motifs.* artifacts untouched
```

## 10. Non-goals

Changing any Phase 01–04 behavior, schema key or `config_hash`; regenerating `motifs.*`;
normalising `S` into `[0, 1]`; comparing scores across different weight profiles or datasets;
per-edge detail files; interactive plots; Phase 05/06 dynamics; a CSV without `--csv`.

