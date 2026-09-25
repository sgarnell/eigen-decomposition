All milestone numbers are now verified (and two errors in the master plan's example output are corrected below). Here is the decomposition.

---

# Phase 04 — decomposed into four sub-phases (04A – 04D)

## A. How I cut it, and why

**Cutting criteria used** (yours, plus two I added because they fit this codebase):

| Criterion | How it drove the cut |
|---|---|
| **Data dependency** | The natural chain is *per-mode structure → across-mode structure → partition → graph*. Each layer consumes only the layer before it, so sub-phases can be implemented and verified strictly in order. |
| **Logical breaks** | Each sub-phase answers exactly one question: "what is each mode made of?" (04A), "do modes share motifs?" (04B), "which neurons form populations?" (04C), "how do populations couple?" (04D). |
| **Simplicity** | Every sub-phase is one module, one CLI increment, one test-doc. No sub-phase requires the whole master plan in mind. |
| **Unification of test results** | Each sub-phase owns a *disjoint* set of test concerns and — critically — a **disjoint set of reference numbers**, so a later sub-phase never has to edit an earlier sub-phase's test. This is enforced by §B.3. |
| **Input/output relation** | Every sub-phase has the same input (`eigen[.<variant>].json`) and the same output (`motifs.json` + `motifs.npz`), with the payload **populated one layer at a time**. This is possible because the schema is frozen in 04A (§B.1) — no schema migration between sub-phases. |
| **Repository convention** (mine) | The project's whole architecture is "one module reads one artifact, writes one artifact, has one CLI, one test pair". Sub-phases keep that: same module, same CLI, same artifacts, monotonically growing. |
| **Risk/size balance** (mine) | 04A ≈ 40 % of the work, 04B ≈ 12 %, 04C ≈ 22 %, 04D ≈ 26 %. 04B is deliberately a small, fast, low-risk win between the two big ones. |

**Explicitly rejected alternative:** four separate modules with four chained artifacts (`motifs.json` → `tracking.json` → `groups.json` → `pathway.json`). It gives even cleaner isolation, but it would change the architecture you already approved, give Phase 05 four files to load, and quadruple the CLI/loader/README boilerplate. The staged-payload design below gets the same isolation without any of that.

## B. Cross-cutting contract (applies to all four sub-phases)

### B.1 The schema is frozen in 04A
`src/clustering/functional_motifs.py` declares **every** constant key set in 04A — `TOP_LEVEL_KEYS`, `PROVENANCE_KEYS`, `CONFIG_KEYS`, `NEURON_PARTICIPATION_KEYS`, `MOTIF_KEYS`, `MEMBER_KEYS`, `LINK_KEYS`, `FAMILY_KEYS`, `OCCURRENCE_KEYS`, `GROUP_KEYS`, `PATHWAY_KEYS`, `NODE_KEYS`, `EDGE_KEYS`, `TOP_MODE_KEYS`, `METADATA_KEYS`, `SAVE_DATA_KEYS` — and `validate_motif_payload_schema` validates **all of them**, including link/family/group/pathway entry shapes, from 04A onward. Later sub-phases never touch these constants or the validator. This is what removes schema churn.

### B.2 Staged-field policy
Fields owned by a later sub-phase are emitted as their **neutral value** and are then *replaced*, never added/removed/renamed/retyped:

| Owner | `motifs.json` state before / after |
|---|---|
| 04A | `motifs` populated; `links: []`; `families` = **k singletons** (every mode is its own family, so `family_of_mode` is always defined); `groups: []`; `pathway: {nodes: [], edges: [], n_nodes: null, n_edges: null, ...}` |
| 04B | `links` populated; `families` merged per stable/flipped links |
| 04C | `groups` populated (total partition) |
| 04D | `pathway` populated (`nodes`, `edges`, `n_nodes`, `n_edges`) |

**Neutral-value convention:** `[]` for collections, `null` for not-yet-computed numbers/counters, `null` for not-yet-computed per-entry values. Schema tests therefore assert *key presence plus a type set that includes the neutral value* (e.g. `isinstance(metadata["n_groups"], (int, type(None)))`), so they pass unchanged at every stage.

`config` and `hash_fields` are complete in 04A (all 26 fields, hashed) — so **`config_hash` is identical at every stage** and a non-default config gets its filename segment from 04A onward.

`motifs.npz` keys are **strictly additive** (04A knows a verification table `key → expected array`; later sub-phases append entries to it). A `stage` array `(1,) <U8` (`"04A"`…`"04D"`) records which sub-phase wrote the cache, and `provenance.stage` records the same for the JSON. No placeholder arrays are ever written.

### B.3 Number-ownership rule (the anti-clash rule)
Each sub-phase owns a **disjoint** block of reference numbers (§I). Two consequences:
* a number that *changes* across sub-phases (e.g. `n_families`: `k` in 04A → `18` in 04B) is **never asserted by the earlier sub-phase**; the earlier sub-phase asserts the *structural invariant* instead (families partition all modes), and only tests the pure helper that produces its own preliminary value;
* later sub-phases may only **add** tests. Editing an existing test is a red flag that the cut was wrong (the one legitimate exception: a test that asserted a neutral value, e.g. `assert payload["links"] == []` — which is why 04A tests assert `isinstance(payload["links"], list)`, not emptiness).

### B.4 Allowed vs forbidden edits per sub-phase
**Allowed:** add new constants/functions/dataclasses to `functional_motifs.py`; add modules to `src/clustering/__init__.py`'s lazy re-export list; add options to `build_argument_parser`; add branches to `main`; populate neutral payload fields; add entries to the npz verification table; extend `render_statistics`/`render_summary_box` with new lines; add tests; extend the README Phase 04 section.
**Forbidden:** change an existing function's signature or semantics; rename/retype/remove a schema key or config field; modify `src/utils/io.py`, `src/parsing/*`, `src/matrices/*`, `src/spectral/*`, `pytest.ini`, `environment.yml`, or any Phase 01–03 test/artifact.

### B.5 Verification protocol run at the end of every sub-phase
```bash
PY=/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python
$PY -m pytest -m "not slow" -q                     # fast suite green
$PY -m pytest -q                                   # full suite green
$PY -m src.clustering.functional_motifs \
    -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json \
    -o data/processed --stats                      # reproduces this sub-phase's numbers
git --no-pager diff --stat                         # only expected files touched
git --no-pager status --short                      # no Phase 01-03 artifact touched
```
Plus a `SOURCE_DATE_EPOCH=... ` double-run byte-identity check once artifacts are written (04A onward).

## C. Overview

| | **04A** Participation & motif extraction | **04B** Cross-mode motif tracking | **04C** Functional neuron groups | **04D** Pathway diagram, plots & handoff |
|---|---|---|---|---|
| **Question answered** | what is each mode made of? | do modes share motifs? | which neurons form populations? | how do populations couple? |
| **Input** | `eigen[.v].json` (+ optional `parsed_graph.json` / `--anatomy`) | 04A's `motifs.json` (same input re-read) | same | same (+ optional `z_matrix[.v].json`) |
| **Payload fields populated** | `motifs`, `families` (singletons), `metadata` (motif block), `provenance` | `links`, `families` (merged), `metadata` (link block) | `groups`, `metadata` (group block) | `pathway`, `metadata` (pathway block) |
| **New artifacts** | `motifs.json`, `motifs.npz`, `motifs.data.json` | — (same files, richer) | — | `motifs.pathway.graphml` (PNG diagrams dropped in favour of GraphML) |
| **New CLI flags** | input/threshold/participation/anatomy + `--config-hash --no-sidecar --save-data --stats --strict --dry-run` | `--family-jaccard --family-polarity --link-min-jaccard` | `--grouping --polarity-split --polarity-min-members --max-group-size --n-groups --linkage --affinity --merge-threshold` | `--pathway-source --pathway-weight --pathway-edge-threshold --pathway-top-edges --no-intra --graphml/--no-graphml` |
| **New deps used** | none (numpy) | none (numpy) | scikit-learn (opt-in only) | networkx (GraphML backend; `--plot`/matplotlib not used) |
| **New tests (target)** | ≈52 (+2 slow) | ≈18 (+1 slow) | ≈30 (+2 slow) | ≈32 (+2 slow) |
| **Cumulative suite** | 397 → ≈449 | → ≈467 | → ≈497 | → ≈536 |
| **Size / risk** | large / medium | small / low | medium / low-medium | medium / medium |

**see execution plan 04A for details**

## E. Sub-phase 04B — Cross-mode motif tracking (links & families)

**Objective.** Decide whether and how modes share subcircuits: the pairwise link table and the merged motif families.

**Prerequisite.** 04A green.

**See document 04B_tracking.md for details**

## F. Sub-phase 04C — Functional neuron groups

**Objective.** Reduce `N` neurons to `G` population variables grounded in motif membership (and, opt-in, loading-vector similarity).

**Prerequisite.** 04B green.

**See execution plan document 04C_grouping.md for details**

## G. Sub-phase 04D — Pathway diagram, figures & Phase 05 handoff

Pathway construction (GraphML output)
The final analysis layer builds a directed functional pathway graph between neuron groups.
Edges represent signed mode contributions aggregated across motifs, optionally enriched with raw matrix contributions when available.
The result is exported as a GraphML document (motifs.pathway.graphml) containing:

* group nodes with full metadata
* weighted directed edges
* polarity and sign information
* per‑mode contribution breakdown
* optional raw Z‑matrix contributions
* threshold and top‑N flags
* intra‑group vs cross‑group classification

This GraphML artifact is designed for interactive exploration in yEd, enabling:
* custom layouts
* edge routing
* node grouping
* metadata inspection
* recoloring and styling
* exporting to SVG/PNG/PDF
* manual refinement for publication‑quality figures

**See execution plan 04D_pathways.md for the spec and `04D_pathways_detailed.md` for the
implemented design.** Implemented defaults on the reference dataset: 22 nodes / 39 edges,
`abs_max = 2.345300`, smallest kept `|edge| = 0.235975`, 0 positive / 39 negative edges,
7 intra / 32 cross edges, `weight_concentration = 0.055842`, threshold-only sweep
`91 / 46 / 17 / 14` at `0.05 / 0.1 / 0.2 / 0.25 × abs_max`. The spec's `--plot`,
`--plot-pathways`, `--no-popup`, `--cmap`, `--max-diagram-nodes` and `--diagram-layout`
flags were never implemented (the GraphML artifact replaces the PNG layer); the frozen
config fields `max_diagram_nodes` / `diagram_layout` remain declared and unused.

## H. Schema ownership table (who populates what — frozen in 04A)

| Payload block / key group | Declared | `n` after 04A | after 04B | after 04C | after 04D |
|---|---|---|---|---|---|
| `provenance` (+ `stage`) | 04A | ✔ | stage→B | stage→C | stage→D |
| `config` (all 26 fields, hashed) | 04A | ✔ | — | — | — |
| `neuron_order` | 04A | ✔ | — | — | — |
| `motifs[]` | 04A | ✔ | — | — | — |
| `links[]` | 04A | `[]` | ✔ | — | — |
| `families[]` | 04A | k singletons | merged | — | — |
| `groups[]` | 04A | `[]` | `[]` | ✔ | — |
| `pathway.nodes/edges` | 04A | `[]` | `[]` | `[]` | ✔ |
| `pathway.n_nodes/n_edges` | 04A | `null` | `null` | `null` | int |
| `metadata` motif block | 04A | ✔ | — | — | — |
| `metadata` link/family block | 04A | `null` | ✔ | — | — |
| `metadata` group block | 04A | `null` | `null` | ✔ | — |
| `metadata` pathway block | 04A | `null` | `null` | `null` | ✔ |
| `motifs.data.json` key set | 04A | full, partial values | richer | richer | complete |
| `motifs.npz` keys | 04A | 16 keys | +2 | +3 | +3 |

## I. Number-ownership map (the no-clash guarantee)

| Sub-phase | Owns these assertions | Must **not** assert |
|---|---|---|
| 04A | member counts, `n_members_total`, min/median/max/mean, unique/none, recurrence histogram, top recurrent, background neuron, 31 regions, partition invariant of `families`, schema | `n_families` value, `n_links`, anything group/pathway |
| 04B | `n_links`, class counts, `n_families = 18`, multi-mode families | group/pathway numbers |
| 04C | `n_groups = 22`, sizes, background/singleton counts, coherence | pathway numbers |
| 04D | `n_pathway_nodes/edges`, `pathway_abs_max`, 0/39 polarity, intra/cross, concentration, edge-count sweep | — |

## J. Risks of the decomposition, and mitigations

| Risk | Mitigation |
|---|---|
| **Schema churn** between sub-phases (the classic failure of incremental plans) | §B.1 freeze + §B.2 staged-field policy + schema tests written against type sets that include the neutral value |
| **A later sub-phase breaking an earlier test** (e.g. `n_families` 21→18) | §B.3 number-ownership map; pure helpers (`initial_families`, `merge_families`) unit-tested separately from the pipeline |
| An intermediate artifact looking "broken" (empty `groups`, null counters) | §B.2 neutral-value convention is documented in the README and in `provenance.stage`; `--stats` prints "not computed yet (04C)" lines instead of zeros |
| A sub-phase accidentally implementing a later feature | CLI flags for a layer are added only with that layer, so a premature feature is unreachable from the command line and untestable |
| 04A is still large (40 %) | 04A has no plotting, no sklearn, no networkx and no dependency on later layers, so it is long but shallow; if it still feels heavy, split it as `04A-1` (participation + motifs, JSON only) and `04A-2` (npz/save-data/CLI/stats/schema tests) — say the word and I'll cut it that way |
| Later sub-phases quietly changing semantics | the verification protocol (§B.5) includes `git diff --stat` + a byte-identity check, and the previous sub-phase's `motifs.json` can be regenerated and compared field-by-field |

## K. Corrections to the master plan found while verifying numbers

Two example values in the master plan's terminal-output section were illustrative and are now wrong; they must be fixed when 04A/04D are written:

1. `pathway polarity` is **0 positive / 39 negative (7 intra-group)** — **not** "24 positive / 15 negative (8 intra)". At default thresholds *every* kept edge is negative. This is now a documented finding, not a bug: it is consistent with Phase 02's negative-weight bias (77 % of unified weights negative, `mean w = −0.096`, Phase 02 §21) and with the sign convention making mode 1's outer product negative on its top pairs.
2. `n_singleton_groups` is **3** (non-background groups of size 1), plus the background group; `mean_group_size` is **5.136364**; `largest_group` = 10.

Everything else in the master plan's §16 was verified and stands.