
# Phase 02 Update A — Self‑Loop Support Specification  
*(for `src/matrices/build_square_matrix.py`)*

## Purpose
Extend Phase 02 so that **self‑loops in the Phase 01 artifact are allowed** when explicitly requested, instead of always being treated as validation errors.

Phase 02 currently rejects any `(source == target)` pair with:

```
CODE_SELF_LOOP
```

This prevents building a Z‑matrix for datasets that legitimately contain auto‑connections.

Phase 02 must be updated to:

- accept self-loops when a new flag is provided  
- compute unified weights normally  
- place self-loop weights on the diagonal  
- update metadata accordingly  
- preserve existing strict/warning semantics  

---

## Requirements

### 1. New CLI flag
Add a boolean flag:

```
--allow-self-loops
```

Behavior:

- Default: **False** (current behavior)
- When False: self-loops produce a validation **error** and Phase 02 aborts
- When True: self-loops produce a validation **warning** and are **included** in the Z‑matrix

This mirrors Phase 01 Update A.

---

### 2. Validation rule change
Current code (Phase 02):

```python
if source == target:
    report.error(CODE_SELF_LOOP, f"pairs[{index}] encodes the self-loop {source!r}")
```

Required change:

```python
if source == target:
    if allow_self_loops:
        report.warning(CODE_SELF_LOOP, ...)
    else:
        report.error(CODE_SELF_LOOP, ...)
```

This must be applied in `_collect_pairs()` where pairs are validated.

---

### 3. Self-loop weights must be included in the Z‑matrix
Current behavior:

- self-loops are rejected  
- diagonal entries are forced to zero  
- Z‑matrix cannot represent auto-connections  

New behavior:

- When `--allow-self-loops` is used:
  - compute unified weight `w` normally  
  - store `w` at `Z[i, i]`  
  - store `w` in `matrix_symmetric[i, i]`  
  - include the pair in `pairs`  
  - include the weight in `weight_stats`  
  - include the diagonal entry in `matrix_stats`  

Diagonal entries are no longer forced to zero when self-loops are allowed.

---

### 4. Metadata updates
Phase 02 metadata currently includes:

```
"n_self_loops"
```

This must be updated to:

- count self-loops only when allowed  
- remain zero when self-loops are disallowed  
- reflect the number of diagonal entries populated

Add a new provenance/config field:

```
config["allow_self_loops"] = bool
```

And include it in:

- `CONFIG_KEYS`
- `MATRIX_CONFIG_FIELDS` (so it affects config_hash)
- `DEFAULT_DECISION_FIELDS` (so default runs remain canonical)

---

### 5. README updates
Add to Phase 02 documentation:

#### New option
```
--allow-self-loops
    Allow pairs of the form (A, A). When enabled, self-loops become warnings
    instead of errors and are included in the Z-matrix diagonal.
```

#### New subsection
Explain:

- Phase 01 may produce self-loops  
- Phase 02 normally forbids them  
- Use `--allow-self-loops` to include them  
- Diagonal entries will be non-zero  
- Symmetrization preserves diagonal values  
- Spectral radius may change  

---

### 6. Schema compatibility
Self-loops do **not** require changes to:

- `TOP_LEVEL_KEYS`
- `WEIGHTED_PAIR_KEYS`
- `validate_z_payload_schema`

Diagonal entries are valid floats and already accepted.

---

### 7. Testing requirements

#### Test 1 — default behavior
Input: parsed_graph.json containing `(A, A)`  
Command:

```
build_square_matrix --strict
```

Expected:

- `CODE_SELF_LOOP` error  
- Phase 02 aborts  
- no artifact written  

#### Test 2 — allow-self-loops
Command:

```
build_square_matrix --allow-self-loops
```

Expected:

- `CODE_SELF_LOOP` warning  
- artifact written  
- diagonal entry `Z[i, i] = w`  
- metadata `n_self_loops == 1`  
- config contains `"allow_self_loops": true`  

#### Test 3 — mixed dataset
Input: one normal pair, one self-loop  
Expected:

- both pairs included  
- diagonal entry populated  
- off-diagonal entry populated  

#### Test 4 — symmetric mode
Command:

```
build_square_matrix --allow-self-loops --symmetric
```

Expected:

- diagonal preserved  
- symmetric matrix identical to directed matrix  
- spectral radius computed normally  

#### Test 5 — normalization
Verify that:

- row/col norms include diagonal entries  
- zscore-nonzero includes diagonal entries  
- unit normalization includes diagonal entries  

#### Test 6 — config_hash
Verify:

- config_hash changes when `allow_self_loops` changes  
- canonical artifact remains canonical when flag is absent  

---

## Summary (for the AI patch generator)

Implement:

- new CLI flag `--allow-self-loops`
- conditional downgrade of self-loop errors → warnings
- inclusion of self-loop weights on the diagonal
- metadata updates (`n_self_loops`, config flag)
- README updates
- full test suite updates

No changes to:

- JSON schema  
- unified weight formula  
- symmetrization logic  
- normalization logic  

---
