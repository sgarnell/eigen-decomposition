# Phase 01 Update A — Self‑Loop Support Specification

## Purpose
Extend `src.parsing.parse_graphviz` so that **self‑loops are allowed** when explicitly requested, instead of always being treated as validation errors.

Self‑loops occur when a hub encodes a pair `(A, A)`:

```
A -> "A--A" [label="Zscore = <pre_z>"]
"A--A" -> A [label="Zscore = <post_z>"]
```

These appear in real neural datasets and must be supported.

---

## Requirements

### 1. New CLI flag
Add a boolean flag:

```
--allow-self-loops
```

Behavior:

- Default: **False** (current behavior)
- When False: self-loops produce a validation **error**
- When True: self-loops produce a validation **warning** and are **included** in `pairs`

This matches the existing strict/warning/error pattern.

---

### 2. Validation rule change
Current code:

```python
if pair.source == pair.target:
    report.error(CODE_SELF_LOOP, f"hub {name!r} encodes the self-loop {pair.source!r}")
```

Required change:

- If `allow_self_loops` is **False** → keep as error  
- If `allow_self_loops` is **True** → downgrade to warning

New behavior:

```python
if pair.source == pair.target:
    if allow_self_loops:
        report.warning(CODE_SELF_LOOP, ...)
    else:
        report.error(CODE_SELF_LOOP, ...)
```

---

### 3. Self-loop pairs must be included in the artifact
Currently, self-loops are rejected and the parser aborts.

New behavior:

- When `--allow-self-loops` is used:
  - The pair `(A, A, pre_z, post_z)` **must be appended** to `pairs`
  - Degrees and strengths must include self-loops (already handled automatically in `to_dict`)

No other changes required.

---

### 4. Metadata updates
Add a new metadata field:

```
metadata["allow_self_loops"] = bool
```

This mirrors other Phase 01 provenance fields.

---

### 5. README updates
Add to Phase 01 documentation:

#### New option
```
--allow-self-loops
    Allow hubs of the form "A--A". When enabled, self-loops are warnings
    instead of errors and are included in the collapsed pair list.
```

#### New validation rule
Self-loops are normally forbidden because they violate the hub semantics, but real neural datasets may encode auto-connections. Use `--allow-self-loops` to include them.

---

### 6. Schema compatibility
Self-loops do **not** require any schema changes:

- `pairs` already supports `(source, target)` with identical values
- `raw_edges` already stores hub edges verbatim
- `neurons` degree/strength logic already handles self-loops

No JSON schema changes required.

---

### 7. Testing requirements

#### Test 1 — default behavior
Input:

```
A -> "A--A"
"A--A" -> A
```

Expected:

- Validation error `self_loop`
- Parser aborts
- No artifact written

#### Test 2 — allow-self-loops
Command:

```
parse_graphviz --allow-self-loops
```

Expected:

- Validation warning `self_loop`
- Artifact written
- `pairs` contains `{source: "A", target: "A", pre_z: ..., post_z: ...}`
- `neurons` shows `out_degree[A] = 1`, `in_degree[A] = 1`

#### Test 3 — mixed dataset
Dataset with both normal hubs and self-loops.

Expected:

- Normal hubs behave unchanged
- Self-loops included only when flag is present

---

## Summary (for the AI patch generator)

Implement:

- new CLI flag `--allow-self-loops`
- conditional downgrade of self-loop errors → warnings
- inclusion of self-loop pairs when allowed
- metadata provenance field
- README updates

No changes to:

- JSON schema  
- hub collapse logic  
- neuron degree/strength logic  
