Verified the reference numbers — note the spec's example values were placeholders in a few places (real `cent` split is **112/1**, not 57/56; real pre/post means are **0.3492 / 0.3023**). The rewritten plan below uses your JSON schema exactly, with the real values as acceptance targets.

I'm still in **Plan mode**, so I can't write the file yet — full rewritten content is below. Toggle to **Act mode** and I'll persist it to `execution-plans/01_parse_graphviz.md`.

---

# `execution-plans/01_parse_graphviz.md` (rewritten)

## Phase 01 — Parse Graphviz `.gv` Files

### 1. Goals
Convert raw Graphviz DOT files into **one structured JSON artifact** that is the canonical Phase 01 → Phase 02 boundary. Phase 1 is **purely descriptive**: it structuralizes the hub-encoded file, preserves `pre_z` / `post_z` verbatim in the pairs, and performs **no unification**. `w = s·tanh(a·log(post_z/(pre_z+ε)))` belongs exclusively to Phase 2.

**No CSV files are produced or consumed anywhere in this project.** JSON is the only interchange format.

### 2. Input format contract (empirically verified)
`data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv` — 727 KB:

- Edge lines are single-line: `A -> "A--B" [label="Zscore = <pre>" ...]` and `"A--B" -> B [label="Zscore = <post>" ...]`
- 1355 hubs, each with **exactly one** inbound (pre) and **exactly one** outbound (post) edge. Hub LHS always == pre-edge source; hub RHS always == post-edge target. 1355/1355 ✅
- Real-neuron node statements are `label="<id>\ncent= <value>"` and contain a **literal newline inside the quoted label** (all 113) ⇒ pydot is required; line-based parsing is unsafe
- Hub statements: `label=""` plus cosmetic styling
- No self-loops, no duplicate `(source, target)`, reciprocity 574/1355 (predominantly directed)
- `penwidth ≈ 2×z` (mean 2.0011, sd 0.023) and `color` is a binned palette ⇒ **both cosmetic**; only `label` is authoritative
- Z-scores non-negative in [0.1, 1.0]; `cent ∈ {0.1, 0.2}`

### 3. Inputs
- One or more `.gv` / `.dot` files from `data/raw_dot/` (glob discovery; batch supported).

### 4. Output — the single canonical artifact
```
data/processed/<stem>/parsed_graph.json
```
`<stem>` = source filename without extension (e.g. `FB4Yaffect_FB45_999prePost_001_all`).

### 5. JSON schema
```json
{
  "source_file": "FB4Yaffect_FB45_999prePost_001_all.gv",
  "file_sha256": "<hex>",
  "neurons": [
    { "neuron_id": "FB4Y(EB/NO1)_R_2", "cent": 0.2,
      "out_degree": 12, "in_degree": 9, "pre_strength": 4.8, "post_strength": 3.7 }
  ],
  "pairs": [
    { "source": "FB4Y(EB/NO1)_R_2", "target": "hDeltaH_11_C6_1", "pre_z": 0.8, "post_z": 0.6 }
  ],
  "raw_edges": [
    { "src": "FB4Y(EB/NO1)_R_2", "dst": "FB4Y(EB/NO1)_R_2--hDeltaH_11_C6_1",
      "is_hub": true, "z_score": 0.8, "penwidth": 1.6, "color": "blue" }
  ],
  "metadata": {
    "n_neurons": 113, "n_hubs": 1355, "n_raw_edges": 2710, "n_pairs": 1355,
    "density": 0.107064, "reciprocity": 574,
    "pre_stats":  { "min": 0.1, "mean": 0.3492, "max": 1.0 },
    "post_stats": { "min": 0.1, "mean": 0.3023, "max": 1.0 },
    "cent_distribution": { "0.1": 112, "0.2": 1 },
    "filters": [],
    "parser": "pydot", "parser_version": "4.0.1",
    "created_utc": "2026-09-15T23:00:00Z"
  }
}
```

**Writing rules**
- `json.dump(..., sort_keys=True, indent=2, ensure_ascii=False)` + trailing newline, UTF-8, written atomically (temp file + `os.replace`)
- `neurons` sorted by `neuron_id`; `pairs` sorted by `(source, target)`; `raw_edges` sorted by `(src, dst)`
- `pairs` contains **only** `source, target, pre_z, post_z` — no `weight` key, ever (guards Phase 2's ownership)
- `raw_edges` preserves source order of the file after the deterministic sort above; all raw attributes kept verbatim

**Field semantics**
- `file_sha256` — SHA-256 over the source file's raw bytes (provenance)
- `pre_strength` = Σ `pre_z` over outgoing pairs; `post_strength` = Σ `post_z` over incoming pairs (deliberately *not* the unified weight)
- `density` = directed density `P / (N·(N−1))` = 1355/(113·112) = **0.107064**
- `reciprocity` = **count** of ordered pairs whose reverse pair is also present (574), per the spec
- `cent_distribution` keys are **strings**; Phase 1 stores the *raw source token* from the label (e.g. `"0.1"`) so no float-repr drift
- `filters` = list of applied filter descriptors, e.g. `[{"type": "min_pre", "value": 0.2, "removed": 18}]`; `[]` when none
- `parser` = `"pydot"`, `parser_version` from `importlib.metadata.version("pydot")`
- `created_utc` = RFC 3339 UTC (`...Z`)

**Reproducibility**: `created_utc` is the only non-deterministic field. When `SOURCE_DATE_EPOCH` is set, it is honored so output is byte-identical; the determinism test compares everything except `created_utc`.

### 6. Module layout
```
src/parsing/__init__.py            # exports parse_graphviz_file, ParsedGraph, GraphvizValidationError
src/parsing/parse_graphviz.py      # load → hub collapse → validate → filter → JSON writer → CLI
src/utils/__init__.py
src/utils/io.py                    # outdir creation, atomic JSON write, hashing, UTC stamping
tests/fixtures/*.gv
tests/test_parse_graphviz.py
tests/test_parsed_graph_schema.py
```
**Dependency change:** Phase 1 no longer needs `pandas` — only `pydot` + stdlib (`json`, `hashlib`, `dataclasses`, `pathlib`, `argparse`, `datetime`, `collections`).

### 7. Data structures
```python
@dataclass(frozen=True)
class Neuron:
    neuron_id: str
    cent: float | None
    cent_token: str | None        # raw label token, used for cent_distribution keys

@dataclass(frozen=True)
class NeuronPair:
    source: str; target: str; pre_z: float; post_z: float

@dataclass(frozen=True)
class RawEdge:
    src: str; dst: str; is_hub: bool; z_score: float
    penwidth: float | None; color: str | None

@dataclass
class ParsedGraph:
    source_file: Path
    file_sha256: str
    neurons: list[Neuron]
    pairs: list[NeuronPair]
    raw_edges: list[RawEdge]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: ...   # exact schema, sorted lists
```

### 8. Algorithm
1. **Load** via `pydot.graph_from_dot_file` (mandated; only robust option given embedded newlines). Multiple graphs in one file ⇒ merge, or error in `--strict`.
2. **Normalize** pydot output: strip surrounding quotes from names/labels; parse `Zscore = <float>` tolerantly; ignore `graph`/`node`/`edge` pseudo-elements.
3. **Classify** names: hub ⇔ contains `--`; else real neuron.
4. **Neuron metadata**: split the real-node label on the newline, extract `cent=` as both float and raw token.
5. **Bucket edges per hub**: pre-edges (hub is destination) and post-edges (hub is source).
6. **Collapse**: one `NeuronPair(pre_edge.src, post_edge.dst, pre_z, post_z)` per hub.
7. **Raw edges**: emit every original `(src, dst, z_score, penwidth, color)` with `is_hub`.
8. **Sort** deterministically (neurons, pairs, raw_edges).
9. **Validate** (§9) → **optionally filter** (§10) → **compute metadata** → **write atomically**.
10. **Batch mode** over a directory prints a per-file summary and exits non-zero if any file failed in `--strict`.

**CLI**
```
python -m src.parsing.parse_graphviz \
  --input data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv \
  --outdir data/processed \
  [--glob '*.gv'] [--strict] [--min-pre X] [--min-post X] \
  [--top-k N] [--dry-run] [--log-level INFO]
```

### 9. Validation rules
| Check | Severity |
|---|---|
| Hub has exactly 1 pre + 1 post edge | Error |
| Hub name's `lhs`/`rhs` match the actual pre/post endpoints | Error |
| No self-loops (`source != target`) | Error |
| No duplicate `(source, target)` pair | Error |
| Both pair endpoints ∈ neuron list | Error |
| Hub names never appear in `neurons` | Error |
| All `z_score` finite and ≥ 0 (negatives flagged, not silently accepted) | Error |
| Missing/invalid `cent`, `penwidth`, `color`, unexpected attributes | Warning |
| Schema conformance of the produced dict before writing | Error |

Errors raise `GraphvizValidationError` naming the offending hub/edge, or accumulate into a `ValidationReport` (default lenient collect-and-report; `--strict` escalates warnings).

### 10. Optional filtering (default OFF)
`--min-pre`, `--min-post`, `--top-k` (ranked by `pre_z + post_z`). Filters apply **after** pairing so statistics stay interpretable; every filter is recorded in `metadata.filters` with its removed count.

**The canonical `parsed_graph.json` is never overwritten by a filtered run.** Filtered output goes to `data/processed/<stem>/parsed_graph.filtered.json`; default runs write only `parsed_graph.json`.

### 11. Testing strategy

- __Framework__: `pytest` __9.1.1__ (installed in the `eigen-decomposition` conda env; run as `/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest`). Tests live in `tests/` with a `src/`-layout import of `src.parsing…`, which requires `pythonpath = .` in the pytest config (the default `prepend` import mode otherwise inserts `tests/`, not the project root, into `sys.path`). No plugin dependencies are assumed: `pytest-cov`, `pytest-xdist` and `pytest-mock` are __not__ installed, so coverage collection is out of scope for Phase 1 and all fixtures/monkeypatching use pytest's built-ins. No `jsonschema`/`pydantic` is available either, so JSON schema conformance is checked by a small hand-rolled `assert_schema()` helper rather than a schema-validation library. The `slow` marker must be registered in config (pytest 9 warns on unknown markers) and the suite is best run as `-m "not slow"` for the fast path.

- __Test configuration__ — `pytest.ini` at the project root (the project has no `pyproject.toml`/packaging config yet; this can be migrated later):

  ```ini
  [pytest]
  testpaths = tests
  pythonpath = .
  addopts = -ra --strict-markers --strict-config
  markers =
      slow: integration test against the real 727 KB .gv file (deselect with -m "not slow")
  ```

- __Synthetic fixtures__ (hand-written, no large files in git): `tiny_valid.gv` — 3 neurons / 2 hubs, one real-node label containing a literal newline (regression guard for the parsing hazard); `missing_post_edge.gv`, `mismatched_hub_name.gv`, `self_loop.gv`, `duplicate_pair.gv`, `malformed.gv` — one fixture per validation rule; `no_cent.gv` — the metadata-lenience case.

- __Unit tests__: hub classification; label/quote normalization; pre/post assignment; `cent` float + raw-token extraction; deterministic ordering (two runs ⇒ identical bytes); each validation rule (positive + negative); filter arithmetic and `metadata.filters` accounting; SHA-256 correctness; atomic-write behavior (no partial file left behind on a failed write).

- __Schema tests__ (`tests/test_parsed_graph_schema.py`): all six top-level keys present with no extras; `pairs` keys exactly `{source, target, pre_z, post_z}` (__no `weight`__); `neurons`, `raw_edges` and `metadata` keys exact; nested `pre_stats`/`post_stats` exact; `cent_distribution` keys are strings; `sort_keys=True` round-trip stability; `created_utc` parses as RFC 3339.

- __Determinism test__: with `SOURCE_DATE_EPOCH` fixed, two runs produce __byte-identical__ artifacts (`created_utc` is the only nondeterministic field, and it derives from that variable).

- __Integration / real-file test__ (`@pytest.mark.slow`): the reference file must yield `n_neurons=113`, `n_hubs=1355`, `n_raw_edges=2710`, `n_pairs=1355`, `density≈0.107064`, `reciprocity=574`, `pre_stats.mean≈0.3492`, `post_stats.mean≈0.3023`, `cent_distribution == {"0.1": 112, "0.2": 1}`, and zero validation errors. (Note: freezegun is unavailable, so the fixed timestamp is injected via the `SOURCE_DATE_EPOCH` env var using `monkeypatch.setenv`, not by patching the clock.)

- __CLI tests__: `--strict` exit code, `--dry-run` writes nothing, batch mode over a directory, and a non-zero exit when a fixture fails validation.

- __Commands__: fast path `/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -m "not slow" -q`; full path `/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -q`.


### 12. Risks & mitigations
| Risk | Mitigation |
|---|---|
| pydot ~**9.6 s/file** (same for `nx_pydot`) | `parsed_graph.json` is the cached interchange format; re-parse only when the source SHA-256 changes; a fast tokenizer is a documented future optimization that must pass the identical validation suite |
| Embedded newlines in quoted labels | pydot-only loading + dedicated regression fixture |
| pydot label quoting / 204 distinct `Zscore` templates | tolerant regex + normalization tests |
| pydot/pyparsing version drift | pin in `environment.yml`; record `parser_version` in metadata |
| `created_utc` breaks byte-reproducibility | honor `SOURCE_DATE_EPOCH`; determinism test ignores only that field |
| `color` values in this file are names (`red`, `gold`, `blue`, `chartreuse`, `cyan3`, `lime`, `limegreen`, `orangered2`, `orangered3`), while the schema example shows `"3"` | store **verbatim** as a string, uninterpreted (name or index); no mapping applied |
| `is_hub` is `true` for all 2710 edges in this format | define as "either endpoint is a hub" for forward compatibility with hybrid direct-edge files; documented as trivially true today |
| Multi-file semantics | per-file artifacts only; any cross-file union is a Phase 2 decision, never implied here |

### 13. Acceptance criteria
1. `parse_graphviz_file` on the reference file reproduces `113 / 1355 / 2710 / 1355`, `density 0.107064`, `reciprocity 574`, zero errors.
2. Exactly one output file: `data/processed/<stem>/parsed_graph.json`. **No CSV anywhere.**
3. JSON keys sorted; `neurons`/`pairs`/`raw_edges` deterministically ordered; byte-identical across runs under a fixed `SOURCE_DATE_EPOCH`.
4. `pairs` entries contain only `source, target, pre_z, post_z`.
5. Every validation rule has a fixture that triggers it.
6. Artifact round-trips: `json.load(open("parsed_graph.json"))` then `pg["pairs"]` / `pg["neurons"]` yield exactly the Phase 2 inputs with no post-processing.
7. Full `pytest` run green; batch CLI over `data/raw_dot/` succeeds.

### 14. Deliverables
- `src/parsing/parse_graphviz.py` (+ `__init__.py`)
- `src/utils/io.py` (+ `__init__.py`)
- `tests/fixtures/*.gv`, `tests/test_parse_graphviz.py`, `tests/test_parsed_graph_schema.py`
- Populated `environment.yml` (conda-forge pins incl. `pytest`)
- pytest.ini` alongside the test files: `tests/fixtures/*.gv`, `tests/test_parse_graphviz.py`, `tests/test_parsed_graph_schema.py`, __`pytest.ini`__.
- __`environment.yml`__ (still empty) should pin `pytest>=9.1` under the conda-forge dev/test deps. Optionally add `pytest-cov` there if you want coverage in Phase 1 — I've written the plan to assume it is absent, so no test depends on it.


### 15. Out of scope (explicit non-goals)
Z-matrix construction, neuron index maps, symmetrization/normalization, **pre/post unification**, canonical-dataset thresholding, any spectral analysis. All Phase 2+.

---

### Defaults I'll apply unless you say otherwise (flagged, not blocking)
1. **`color`** stored verbatim as string, uninterpreted (your example shows `"3"`, this file has palette names) — no `rdylgn8` mapping.
2. **`is_hub`** = "either endpoint is a hub" (always `true` in this format; forward-compatible).
3. **Filtered runs** write `parsed_graph.filtered.json` so the canonical artifact is never mutated.
4. **`reciprocity`** stays a count (574), as your schema shows, not a ratio.

I need **Act mode** to create `execution-plans/01_parse_graphviz.md`. Once you toggle, tell me whether to write the plan document alone, or the plan **plus** the code/test scaffolding (`src/utils/io.py`, `src/parsing/parse_graphviz.py`, fixtures, tests, and the populated `environment.yml`).