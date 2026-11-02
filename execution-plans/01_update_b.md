**Robust Hub Repair for Partial Hubs in Phase 01 (Graphviz Parser)**

### **Goal:**  
Prevent python -m src.parsing.parse_graphviz  from failing when a hub has only one valid leg (pre or post), by **auto‑repairing** the hub with a minimal synthetic edge instead of raising a fatal error.

This ensures:

- downstream phases (02–04) do not break  
- graph topology is preserved  
- synthetic repairs are traceable  
- analytical integrity is maintained

---

# **1. Problem Definition**
DOT files for large datasets (FB6, FB7, SMP) often contain **partial hubs**:

```
hub 'A--B' has 1 pre edge(s) and 0 post edge(s)
```

This occurs because upstream filtering (FBL weak‑leg removal, HK merges, ring merges) removes one leg of the hub.

Phase 01 currently treats this as a **fatal error**, causing:

- Phase 01 failure  
- Phase 02 missing pairs  
- Phase 03 incomplete Z‑matrix  
- Phase 04 motif/pathway failures  

This cascades through the entire pipeline.

---

# **2. Proposed Solution**
### **Add a new repair mode: `--allow-stub-edges`**

When enabled:

- If a hub has **exactly one** valid leg (pre or post),  
- The parser **auto‑generates** a synthetic missing leg  
- The hub becomes a valid pair  
- The synthetic edge is marked as such  
- A warning is issued instead of an error

This preserves the pair and prevents downstream breakage.

---

# **3. Behavior Changes**
### **Current behavior (strict):**
If `len(pre) != 1 or len(post) != 1`:

- Emit `hub_degree` error  
- Skip the hub  
- Drop the pair  
- Pipeline breaks downstream

### **New behavior (with `--allow-stub-edges`):**
If `len(pre) == 1 and len(post) == 0`:

- Create synthetic post edge  
- Mark synthetic edge  
- Issue warning  
- Continue

If `len(pre) == 0 and len(post) == 1`:

- Create synthetic pre edge  
- Mark synthetic edge  
- Issue warning  
- Continue

If both are missing:

- Still error (cannot repair)

---

# **4. Synthetic Edge Specification**
Synthetic edges must be:

| Field | Value |
|-------|--------|
| `z_score` | `0.1` (or configurable) |
| `color` | `"grey"` |
| `penwidth` | minimal (e.g., `0.5`) |
| `synthetic` | `True` (new metadata flag) |
| `reason` | `"stub_edge_missing_pre"` or `"stub_edge_missing_post"` |

This ensures:

- downstream phases treat the edge as extremely weak  
- synthetic edges are visually distinct  
- synthetic edges are traceable in metadata  
- analytical impact is minimal

---

# **5. Metadata Additions**
Add to `ParsedGraph.metadata`:

```
"synthetic_edges": [
    {
        "hub": "FB6A--FB6K",
        "missing_leg": "post",
        "z_score": 0.1,
        "color": "grey"
    },
    ...
]
```

This allows:

- auditing  
- debugging  
- downstream filtering if needed

---

# **6. CLI Additions**
Add new flag:

```
--allow-stub-edges
```

Semantics:

- Off by default  
- When enabled, partial hubs are repaired  
- When disabled, current strict behavior remains

---

# **7. Logging / Reporting**
When a stub edge is created:

```
WARNING [stub_edge] hub 'FB6A--FB6K' missing post edge; synthetic stub created (z=0.1, color=grey)
```

This replaces the current fatal error.

---

# **8. Downstream Impact**
### **Phase 02**
- Z‑matrix remains square  
- No missing pairs  
- No broken symmetry

### **Phase 03**
- SVD stable  
- No missing rows/columns  
- No NaNs

### **Phase 04**
- Motifs and pathways complete  
- No missing connectivity  
- Synthetic edges appear as weak contributions

This is exactly what you want.

---

# **9. Summary**
`01_update_b` introduces a robust hub‑repair mechanism that:

- prevents cascading failures  
- preserves graph topology  
- maintains analytical integrity  
- keeps synthetic edges traceable  
- requires minimal code changes  
- is fully backward‑compatible
