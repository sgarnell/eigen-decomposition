# Focused GraphML code review

## Scope

Reviewed only the GraphML-related implementation in:

- `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py`
- `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py`

The relevant implementation is concentrated in lines 310–342, 3492–3645, and the public exports around lines 241–269 of `__init__.py`.

## Executive summary

| Requirement | Assessment |
|---|---|
| Every node has a human-readable label attribute | **Yes** |
| Nodes should receive readable labels on yEd import | **Yes**, because the writer emits a string attribute named `label` |
| Every edge has numeric weight metadata | **Yes** |
| Edges are automatically labeled with their weights in yEd | **No** |
| Weight-related attributes are present for yEd mapping/filtering | **Yes**: `weight`, `abs_weight`, and `polarity` |
| Weights are JSON strings | **No**; the weights are numeric GraphML attributes |
| Per-mode contribution details are JSON strings | **Yes** |
| Edges automatically vary in color or width by weight | **No** |
| Optional unavailable metadata is omitted | **Yes** |
| Omission can leave a normal node unlabeled | **No**, because `label` is mandatory in the current helper |
| Omission can leave an edge without weight metadata | **No**, because `weight` and `abs_weight` are mandatory in the current helper |

The GraphML is a sound **annotated graph for interactive mapping in yEd**, but it is not a pre-styled yEd graph. Nodes should be readable automatically; edge weights are available as metadata but are not automatically rendered as edge labels, line widths, or colors.

---

# 1. Node labels

## Does every GraphML node include a text label attribute?

**Yes.** `GRAPHML_NODE_KEYS` includes `"label"`:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:311-321`
>
> ```python
> GRAPHML_NODE_KEYS = (
>     "group_id",
>     "label",
>     "size",
>     "dominant_mode",
>     "region_composition",
>     "centroid",
>     "coherence",
>     "is_background",
>     "is_singleton",
> )
> ```

More importantly, `graphml_node_metadata` does not merely permit that key—it constructs it unconditionally:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3503-3511`
>
> ```python
> record = dict(group or {})
> metadata: dict[str, Any] = {
>     "group_id": str(node.get("group_id")),
>     "label": str(node.get("label") or node.get("group_id")),
>     "size": int(node.get("size", 0)),
>     "region_composition": json.dumps(node.get("region_composition") or {}, sort_keys=True),
>     "is_background": bool(record.get("is_background", False)),
>     "is_singleton": bool(record.get("is_singleton", False)),
> }
> ```

The result is a GraphML string value:

```python
"label": str(node.get("label") or node.get("group_id"))
```

The fallback behavior is appropriate:

1. It prefers the pathway node’s explicit `label`.
2. If that is empty, it falls back to `group_id`.
3. For normal Phase 04D pathway data, this results in labels such as `"G00"`, `"G01"`, etc., consistent with the Phase 04D design.

The underlying pathway node also declares a label field:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:977-987`
>
> ```python
> class PathwayNode:
>     """A pathway-diagram node (= a functional group; populated by 04D)."""
>
>     node_id: str
>     group_id: str
>     label: str
>     members: tuple[str, ...] = ()
>     cent_mean: float | None = None
>     region_composition: dict[str, int] = field(default_factory=dict)
>     dominant_mode: int | None = None
> ```

And serializes it into the dictionary passed to the GraphML writer:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:993-1003`
>
> ```python
> def to_dict(self) -> dict[str, Any]:
>     return {
>         "node_id": self.node_id,
>         "group_id": self.group_id,
>         "label": self.label,
>         ...
>     }
> ```

## Is the label written in a yEd-compatible way?

**Generally yes.** The writer passes the metadata into a NetworkX node:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3580-3588`
>
> ```python
> graph = nx.DiGraph()
> for node in nodes:
>     node_id = str(node.get("node_id"))
>     graph.add_node(
>         node_id,
>         **graphml_node_metadata(
>             node, groups_by_id.get(str(node.get("group_id"))), centroid_by_id.get(node_id)
>         ),
>     )
> ```

NetworkX then serializes these as GraphML `<data>` elements through:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3641-3645`
>
> ```python
> try:
>     body = "\n".join(nx.generate_graphml(graph))
> except Exception as exc:
>     raise MotifValidationError(f"could not serialise the GraphML pathway: {exc}") from exc
> return _write_text_atomic(path, f"{XML_DECLARATION}\n{body}\n")
> ```

A data field literally named `label` with a string value is the conventional GraphML attribute that yEd can use as the node label. The important distinction is that GraphML itself does not define visual semantics for arbitrary data; `label` is the conventional/imported label field, while attributes such as `size` and `coherence` remain metadata unless mapped to styles.

There is one minor robustness issue: if both `label` and `group_id` were missing, this expression would produce the literal string `"None"`:

```python
str(None or None)
```

That is **not responsible for normal unlabeled nodes**, because valid Phase 04D nodes provide both fields. A stricter implementation would explicitly validate non-empty IDs before conversion.

## Are `group_id` and `label` duplicates?

**Yes, but functionally intentional.**

- `group_id`: semantic group identifier
- `label`: visual node label
- `node_id`: NetworkX/GraphML graph identifier

Because `label` is explicitly populated, yEd does not need to infer it from `node_id` or `group_id`.

## Node-label conclusion

**Nodes should appear with readable labels such as `G00`, `G01`, etc. in yEd.** There is no normal-path omission that removes the node label.

---

# 2. Edge labels

## Does each edge have text metadata that yEd could use?

**Not as an explicit edge-label field.**

The allowed edge keys are:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:322-333`
>
> ```python
> GRAPHML_EDGE_KEYS = (
>     "source_group",
>     "target_group",
>     "weight",
>     "abs_weight",
>     "polarity",
>     "contribution_by_mode",
>     "threshold_flag",
>     "topN_flag",
>     "intra_flag",
>     "z_contribution",
> )
> ```

Each edge receives the following attributes:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3527-3539`
>
> ```python
> metadata: dict[str, Any] = {
>     "source_group": str(edge.get("source")),
>     "target_group": str(edge.get("target")),
>     "weight": float(edge.get("weight", 0.0)),
>     "abs_weight": float(edge.get("abs_weight", 0.0)),
>     "polarity": int(edge.get("polarity", 0)),
>     "contribution_by_mode": json.dumps(
>         [dict(entry) for entry in edge.get("top_modes") or ()], sort_keys=True
>     ),
>     "threshold_flag": True,
>     "topN_flag": True,
>     "intra_flag": bool(edge.get("is_intra", False)),
> }
> ```

These are attached as real GraphML edge data attributes:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3589-3596`
>
> ```python
> for edge in edges:
>     metadata = graphml_edge_metadata(
>         edge,
>         z_contribution=(z_by_edge or {}).get(
>             (str(edge.get("source")), str(edge.get("target")))
>         ),
>     )
>     graph.add_edge(str(edge.get("source")), str(edge.get("target")), **metadata)
> ```

## Does the code write an edge `label`?

**No.** There is no:

```python
"label": ...
```

in `GRAPHML_EDGE_KEYS` or `graphml_edge_metadata`.

The text fields it writes are:

- `source_group`
- `target_group`
- `contribution_by_mode`

The weight fields are numeric:

- `weight`
- `abs_weight`
- `polarity`

## Can yEd display these automatically?

**No, not as visible edge labels.**

GraphML edge data fields are metadata. yEd can import and expose them in its data table, and the user can manually configure an edge-label mapping or visual mapping. However, the mere presence of:

```xml
<key ... attr.name="weight" attr.type="double"/>
...
<data key="...">-2.345299...</data>
```

does not mean yEd will display `-2.345299` as an edge label.

Similarly, `polarity` is not automatically rendered as “negative” or “positive” on the edge. It is an integer metadata field.

The same issue applies to `contribution_by_mode`: it is JSON data, not a rendered label. For example, it may contain entries equivalent to:

```json
[{"mode": 1, "share": 0.99, "weight": ...}]
```

The JSON string is valuable for inspection, filtering, or parsing, but yEd will not automatically turn it into a readable label.

## Edge-label conclusion

The output is **rich in edge metadata**, but it does not produce a conventional display label such as:

```text
G04 → G01: -2.345
```

There is no edge-side `label` field, and there is no yEd visual-label mapping or yFiles style information that turns the existing fields into a visible label.

---

# 3. Weight metadata

## Are edge weights included?

**Yes, comprehensively.**

The signed weight and absolute weight are always included:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3530-3532`
>
> ```python
> "weight": float(edge.get("weight", 0.0)),
> "abs_weight": float(edge.get("abs_weight", 0.0)),
> "polarity": int(edge.get("polarity", 0)),
> ```

The underlying pathway edge also stores the numeric signed weight:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:1008-1016`
>
> ```python
> class PathwayEdge:
>     """A signed pathway edge between two groups (populated by 04D)."""
>
>     source: str
>     target: str
>     weight: float
>     modes: tuple[int, ...] = ()
>     is_intra: bool = False
>     polarity: int = 0
> ```

It serializes both the signed and absolute values:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:1027-1037`
>
> ```python
> def to_dict(self) -> dict[str, Any]:
>     return {
>         "source": self.source,
>         "target": self.target,
>         "weight": float(self.weight),
>         "abs_weight": float(self.abs_weight),
>         "n_modes": int(self.n_modes),
>         "modes": [int(mode) for mode in self.modes],
>         "is_intra": bool(self.is_intra),
>         "polarity": int(self.polarity),
>         "top_modes": [dict(entry) for entry in self.top_modes],
>     }
> ```

## Are weights numeric or JSON strings?

They are **native numeric GraphML attributes**, not JSON strings.

That follows directly from the casts:

```python
"weight": float(...)
"abs_weight": float(...)
"polarity": int(...)
```

NetworkX will infer and emit their GraphML types as floating-point or integer values. The implementation does not wrap them with `json.dumps`.

The only structured edge field encoded as JSON is:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3533-3535`
>
> ```python
> "contribution_by_mode": json.dumps(
>     [dict(entry) for entry in edge.get("top_modes") or ()], sort_keys=True
> ),
> ```

Therefore:

- `weight`: numeric
- `abs_weight`: numeric
- `polarity`: numeric
- `z_contribution`: numeric when available
- `contribution_by_mode`: JSON string
- `source_group`, `target_group`: strings
- `threshold_flag`, `topN_flag`, `intra_flag`: booleans

One naming nuance: `contribution_by_mode` is a JSON representation of the edge’s `top_modes`, not necessarily every mode. The detailed Phase 04D design says these are the top five signed mode contributions.

## Is there any yEd-specific weight styling?

**No.**

I found no requested GraphML code in this module for any of the following concepts:

- edge label
- edge line width
- edge thickness
- edge color
- positive/negative color assignment
- yEd visual mapping
- yFiles style descriptions
- GraphML visual properties

The entire edge allowlist is:

```python
GRAPHML_EDGE_KEYS = (
    "source_group",
    "target_group",
    "weight",
    "abs_weight",
    "polarity",
    "contribution_by_mode",
    "threshold_flag",
    "topN_flag",
    "intra_flag",
    "z_contribution",
)
```

None is a visual style key.

Furthermore, the writer explicitly rejects keys outside this allowlist:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3629-3634`
>
> ```python
> for source_id, target_id, attributes in graph.edges(data=True):
>     unknown = sorted(set(attributes) - set(GRAPHML_EDGE_KEYS))
>     if unknown:
>         raise MotifValidationError(
>             f"GraphML edge {source_id}->{target_id} carries unknown keys {unknown}"
>         )
> ```

Thus, adding a hypothetical attribute such as `edge_color` or `edge_width` inside `graphml_edge_metadata` would currently fail validation unless `GRAPHML_EDGE_KEYS` were also expanded.

## What is missing?

There are two possible implementation levels.

### 1. Data-level enhancement

To create a directly labelable edge field, the writer could add an edge-side string attribute such as:

```python
"label": f"{weight:.4g} ({polarity:+d})"
```

That would require:

1. Adding `"label"` to `GRAPHML_EDGE_KEYS`.
2. Adding it to `graphml_edge_metadata`.

That would make the label available as GraphML data, but whether yEd displays an imported edge `label` automatically can depend on yEd’s importer behavior/version. It would still be safer to provide a yEd/yFiles visual-label style or provide a documented post-import mapping.

### 2. Complete yEd visual styling

For automatic weight-dependent width and color, the GraphML would need yFiles/yEd-compatible visual information, for example:

- a label style using `weight` or a preformatted label;
- a line-width style calculated from `abs_weight`;
- a line-color style based on `polarity`;
- possibly a label placement or edge grouping style.

That logic does not currently exist.

## What already supports manual yEd mapping?

The current data is adequate for a user to configure mappings after import:

- map `abs_weight` to edge width;
- map `polarity` to red/blue or green/red color;
- map `weight` or an added label field to edge label;
- filter on `threshold_flag`, `topN_flag`, and `intra_flag`.

So the current artifact is useful for **interactive exploration and manual styling**, but not pre-styled.

---

# 4. Missing attributes and `None` handling

## Does the code omit `None` attributes?

**Yes, for optional node and edge attributes.**

For nodes:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3512-3519`
>
> ```python
> dominant_mode = node.get("dominant_mode")
> if dominant_mode is not None:
>     metadata["dominant_mode"] = int(dominant_mode)
> if centroid is not None:
>     metadata["centroid"] = json.dumps([float(value) for value in np.asarray(centroid).ravel()])
> coherence = record.get("coherence")
> if coherence is not None:
>     metadata["coherence"] = float(coherence)
> ```

For edges:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3540-3541`
>
> ```python
> if z_contribution is not None:
>     metadata["z_contribution"] = float(z_contribution)
> ```

This is a necessary design choice because NetworkX GraphML cannot serialize arbitrary `None` attribute values as ordinary typed data.

## Could omitting optional node attributes make nodes unlabeled?

**No.** The omitted fields are:

- `dominant_mode`
- `centroid`
- `coherence`

The actual label is always inserted earlier:

```python
"label": str(node.get("label") or node.get("group_id"))
```

Therefore omission of optional metadata does not remove the visual label.

## Could omitting optional edge attributes make an edge appear unlabeled?

**No, in the narrower sense of edge-label metadata**, because there is no dedicated edge `label` to begin with. The mandatory fields are always present:

```python
"weight": float(...)
"abs_weight": float(...)
"polarity": int(...)
```

Only `z_contribution` can be omitted.

The important finding is not that omission accidentally deletes `weight`; it is that **the implementation never supplies a displayable edge label at all**.

## Are there any hidden `None` fallbacks?

Most unavailable values are converted to empty/fallback values:

```python
"label": str(node.get("label") or node.get("group_id"))
"size": int(node.get("size", 0))
"region_composition": json.dumps(node.get("region_composition") or {}, ...)
"weight": float(edge.get("weight", 0.0))
"abs_weight": float(edge.get("abs_weight", 0.0))
"polarity": int(edge.get("polarity", 0))
```

This prevents `None` from reaching NetworkX, but it also masks malformed edge records by defaulting absent weights to `0.0`. Given pathway validation earlier in the construction pipeline, this is probably not an active problem, but strict GraphML serialization would be better served by rejecting missing mandatory values rather than silently substituting zero.

---

# 5. Graph-level metadata

The GraphML graph receives these values:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3597-3623`
>
> ```python
> graph.graph.update(
>     {
>         "n_nodes": int(len(nodes)),
>         "n_edges": int(len(edges)),
>         "weight_normalization": PATHWAY_WEIGHT_NORMALIZATION,
>         "pathway_source": str(
>             source if source is not None else payload.get("source") or DEFAULT_PATHWAY_SOURCE
>         ),
>         "pathway_weight_rule": str(
>             config.pathway_weight
>             if config is not None
>             else payload.get("weight") or DEFAULT_PATHWAY_WEIGHT
>         ),
>         "threshold": float(
>             config.pathway_edge_threshold
>             if config is not None
>             else (edge_threshold if edge_threshold is not None else DEFAULT_PATHWAY_EDGE_THRESHOLD)
>         ),
>         "topN": int(
>             config.pathway_top_edges
>             if config is not None
>             else (top_edges if top_edges is not None else DEFAULT_PATHWAY_TOP_EDGES)
>         ),
>     }
> )
> ```

This metadata is correct and useful for inspection, especially:

- `weight_normalization`
- `pathway_source`
- `pathway_weight_rule`
- `threshold`
- `topN`

However, graph metadata is not a display directive. It does not tell yEd to map weights to visual properties.

---

# 6. Public API consistency

The public exports in `src/clustering/__init__.py` correctly include the requested symbols and constants:

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py:181-183`
>
> ```python
> "GRAPHML_EDGE_KEYS",
> "GRAPHML_GRAPH_KEYS",
> "GRAPHML_NODE_KEYS",
> ```

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py:241-242`
>
> ```python
> "graphml_edge_metadata",
> "graphml_node_metadata",
> ```

> `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py:269`
>
> ```python
> "write_graphml_pathway",
> ```

There is no `__init__.py` defect affecting the GraphML labels or weights. It correctly re-exports the implementation.

---

# Final assessment

## Should nodes appear with readable labels in yEd?

**Yes.**

Each node receives an unconditional string `label` attribute, with a `group_id` fallback:

```python
"label": str(node.get("label") or node.get("group_id"))
```

On the reference pathway, this should produce labels such as `G00` through `G21`.

## Should edges appear with readable labels in yEd?

**Not automatically.**

The edges contain usable data attributes:

```python
"weight": float(...)
"abs_weight": float(...)
"polarity": int(...)
```

but there is:

- no edge `label`;
- no preformatted textual weight;
- no yEd/yFiles label style;
- no automatic label mapping.

Therefore, the edge weights can be inspected and manually mapped in yEd, but users should not expect the numerical weights to appear next to edges immediately after import.

## Should edges visually reflect their weights?

**No—not without post-import mapping or additional generation logic.**

The writer has no logic that:

- converts `abs_weight` into edge width;
- maps `polarity` to color;
- adds a weight-based legend;
- embeds yFiles visual properties.

The current artifact is metadata-rich, but visually neutral with respect to weight.

## Exact missing logic

The missing logic belongs in:

1. `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3523-3542`  
   `graphml_edge_metadata`: no display `label` or styling fields.

2. `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:322-333`  
   `GRAPHML_EDGE_KEYS`: no edge-label or style keys.

3. `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py:3580-3645`  
   `write_graphml_pathway`: builds and serializes a plain NetworkX graph, with no yEd/yFiles style construction.

For a data-only improvement, add a textual edge `label` derived from `weight` and `polarity`. For true automatic color/width rendering, add yFiles/yEd-compatible visual styles based on `abs_weight` and `polarity`, and update the GraphML edge-key validation accordingly.

## Overall verdict

- **Correct human-readable node labels:** Yes.
- **Correct numeric weight metadata:** Yes.
- **Correct JSON per-mode contribution metadata:** Yes, representing the top-mode breakdown.
- **Automatic yEd node visibility labels:** Yes.
- **Automatic yEd edge weight labels:** No.
- **Automatic weight-based yEd styling:** No.
- **Omission-caused unlabeled nodes or weightless edges:** No.
- **Suitable for yEd interactive manual mapping:** Yes.
- **Suitable as a publication-ready pre-styled yEd file:** No.

---
# Detailed Phase 04D GraphML patch plan

## Scope and implementation strategy

Limit production changes to:

- `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py`
- `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py`

Tests and documentation require corresponding updates outside those two files. The implementation should remain a **GraphML-only enhancement**:

- Do not modify the canonical pathway payload.
- Do not modify `motifs.json`, `motifs.data.json`, or `motifs.npz`.
- Do not add CLI/config fields.
- Do not alter `MOTIF_CONFIG_FIELDS` or `config_hash`.
- Do not touch Phase 01–03 code.
- Keep NetworkX as the GraphML generator, then add the yFiles extension after NetworkX serialization.

The key implementation constraint is that NetworkX cannot directly create nested yFiles XML such as `<y:EdgeLabel>` from a normal graph attribute. Therefore, the writer should:

1. Build the graph and ordinary scalar attributes with NetworkX.
2. Serialize it with `nx.generate_graphml()`.
3. Parse the generated XML with `xml.etree.ElementTree`.
4. Insert the yFiles namespace, edge-graphics key, and one style block per edge.
5. Serialize the augmented document through the existing atomic writer.

---

# 1. Proposed final GraphML contract

## Node data keys

Keep the existing keys unchanged:

```python
GRAPHML_NODE_KEYS = (
    "group_id",
    "label",
    "size",
    "dominant_mode",
    "region_composition",
    "centroid",
    "coherence",
    "is_background",
    "is_singleton",
)
```

No new node field is strictly required. Existing `size` and `coherence` are sufficient for yEd filtering and manual node-style mapping.

## Edge data keys

Change:

```python
GRAPHML_EDGE_KEYS = (
    "source_group",
    "target_group",
    "label",
    "weight",
    "abs_weight",
    "polarity",
    "contribution_by_mode",
    "threshold_flag",
    "topN_flag",
    "intra_flag",
    "z_contribution",
    "weight_ratio",
    "edge_width",
    "edge_color",
)
```

New fields:

| Field | Type | Purpose |
|---|---|---|
| `label` | string | Human-readable edge label such as `G04 → G01: -2.3453` |
| `weight_ratio` | double | `abs_weight / abs_max`, in `[0, 1]`; useful for filtering |
| `edge_width` | double | Concrete yFiles line width derived from `weight_ratio` |
| `edge_color` | string | Concrete hex color derived from `polarity` |

`weight`, `abs_weight`, `polarity`, and optional `z_contribution` must remain numeric.

`contribution_by_mode` must remain a deterministic JSON string.

## Graph-level keys

Extend:

```python
GRAPHML_GRAPH_KEYS = (
    "n_nodes",
    "n_edges",
    "abs_max",
    "weight_normalization",
    "edge_style",
    "pathway_source",
    "pathway_weight_rule",
    "threshold",
    "topN",
)
```

New fields:

| Field | Type | Purpose |
|---|---|---|
| `abs_max` | double | Scale used to calculate edge width |
| `edge_style` | JSON string | Documents the width and color formulas for reproducibility |

The `edge_style` value should resemble:

```json
{
  "color_by_polarity": {
    "-1": "#C62828",
    "0": "#757575",
    "1": "#1565C0"
  },
  "width_formula": "1.0 + 4.0 * min(1.0, abs_weight / abs_max)",
  "width_max": 5.0,
  "width_min": 1.0
}
```

This graph metadata does not replace the yFiles style blocks; it documents how those blocks were produced.

---

# 2. File-by-file plan

## A. `/home/mr-miracle/projects/eigen-decomposition/src/clustering/functional_motifs.py`

### A1. Add the XML namespace import

At the current import block around lines 80–93, add:

```python
from xml.etree import ElementTree as ET
```

No external dependency is needed.

### A2. Add private GraphML styling constants

Immediately after `XML_DECLARATION` around current line 309, add:

```python
#: Namespace used by the yFiles/yEd GraphML visual-properties extension.
YFILES_GRAPHML_NAMESPACE = "http://www.yworks.com/xml/graphml"

#: Fixed style key for per-edge yFiles edge graphics.  A human-readable fixed ID is
#: intentional: the existing deterministic GraphML tests compare complete documents.
GRAPHML_EDGE_STYLE_KEY = "yfiles_edge_graphics"

#: yFiles line width is expressed in points.
GRAPHML_EDGE_WIDTH_MIN = 1.0
GRAPHML_EDGE_WIDTH_MAX = 5.0

#: Colour convention: negative = red, positive = blue, neutral = gray.
GRAPHML_EDGE_COLORS = {
    -1: "#C62828",
    0: "#757575",
    1: "#1565C0",
}
```

These do not need to be added to `__all__`; they can be treated as implementation constants. Alternatively, if the project’s convention requires all uppercase GraphML constants to be public, they can be exported from `__init__.py`, but private underscore names would reduce API churn.

### A3. Extend the GraphML key sets

At current lines 310–342, make the changes shown in Section 1.

The additions to `GRAPHML_EDGE_KEYS` must occur before the writer validates `graph.edges(data=True)`. Otherwise the new fields would be rejected by the existing unknown-key logic.

### A4. Add robust text validation helper

Add before `graphml_node_metadata`, around current line 3492:

```python
def _required_graphml_text(
    value: Any,
    *,
    field: str,
    context: str,
) -> str:
    """Return a non-empty GraphML text value or raise a contextual error."""
    if value is None:
        raise MotifValidationError(f"{context} {field!r} must not be None")
    text = str(value).strip()
    if not text or text == "None":
        raise MotifValidationError(
            f"{context} {field!r} must be a non-empty string other than 'None'"
        )
    return text
```

This fixes the current behavior:

```python
str(node.get("group_id"))
```

which converts a missing ID into the literal `"None"` rather than rejecting it.

### A5. Add edge-label and style-calculation helpers

Add between the text validator and `graphml_edge_metadata`:

```python
def _graphml_edge_label(
    source: str,
    target: str,
    weight: float,
) -> str:
    """Return a stable human-readable label for a signed pathway edge."""
    formatted_weight = "0.0000" if weight == 0.0 else f"{weight:+.4f}"
    return f"{source} → {target}: {formatted_weight}"


def _graphml_edge_width(abs_weight: float, abs_max: float) -> tuple[float, float]:
    """Return ``(normalized_ratio, yfiles_width)`` for an edge."""
    if abs_max > 0.0 and math.isfinite(abs_max):
        ratio = min(1.0, max(0.0, abs(float(abs_weight)) / abs_max))
    else:
        ratio = 0.0
    width = GRAPHML_EDGE_WIDTH_MIN + (
        GRAPHML_EDGE_WIDTH_MAX - GRAPHML_EDGE_WIDTH_MIN
    ) * ratio
    return ratio, width
```

Recommended formula:

```text
weight_ratio = min(1, abs_weight / abs_max)
edge_width  = 1.0 + 4.0 * weight_ratio
```

This provides:

- minimum width: `1.0`
- maximum width: `5.0`
- zero/abs-max-zero fallback: `1.0`
- deterministic behavior independent of yEd

### A6. Harden `graphml_node_metadata`

Replace current lines 3503–3511 with logic equivalent to:

```python
    context = f"GraphML node {node.get('node_id', '<unknown>')!r}"
    group_id = _required_graphml_text(
        node.get("group_id"),
        field="group_id",
        context=context,
    )

    raw_label = node.get("label")
    if raw_label is None or str(raw_label).strip() in {"", "None"}:
        raw_label = group_id
    label = _required_graphml_text(
        raw_label,
        field="label",
        context=context,
    )

    record = dict(group or {})
    metadata: dict[str, Any] = {
        "group_id": group_id,
        "label": label,
        "size": int(node.get("size", 0)),
        "region_composition": json.dumps(
            node.get("region_composition") or {},
            sort_keys=True,
        ),
        "is_background": bool(record.get("is_background", False)),
        "is_singleton": bool(record.get("is_singleton", False)),
    }
```

Keep the current optional-field behavior for:

```python
    dominant_mode = node.get("dominant_mode")
    if dominant_mode is not None:
        metadata["dominant_mode"] = int(dominant_mode)

    if centroid is not None:
        metadata["centroid"] = json.dumps(
            [float(value) for value in np.asarray(centroid).ravel()]
        )

    coherence = record.get("coherence")
    if coherence is not None:
        metadata["coherence"] = float(coherence)
```

Behavior:

- A missing or empty node label falls back to a validated `group_id`.
- A missing or invalid `group_id` raises `MotifValidationError`.
- The literal output `"None"` is impossible for normal node IDs or labels.
- Optional descriptive metadata remains omitted when unavailable.

### A7. Change `graphml_edge_metadata` signature and implementation

The writer needs `abs_max` to compute a scale-independent width and `weight_ratio`.

Replace the current helper with:

```python
def graphml_edge_metadata(
    edge: dict[str, Any],
    *,
    abs_max: float = 0.0,
    z_contribution: float | None = None,
) -> dict[str, Any]:
    """Return scalar and yEd-facing GraphML attributes for one pathway edge."""
    source = _required_graphml_text(
        edge.get("source"),
        field="source",
        context="GraphML edge",
    )
    target = _required_graphml_text(
        edge.get("target"),
        field="target",
        context=f"GraphML edge {source!r}->{target!r}",
    )

    weight = float(edge.get("weight", 0.0))
    abs_weight = float(edge.get("abs_weight", abs(weight)))
    polarity = int(edge.get("polarity", int(np.sign(weight))))

    if not math.isfinite(weight) or not math.isfinite(abs_weight):
        raise MotifValidationError(
            f"GraphML edge {source!r}->{target!r} has a non-finite weight"
        )

    weight_ratio, edge_width = _graphml_edge_width(abs_weight, abs_max)

    metadata: dict[str, Any] = {
        "source_group": source,
        "target_group": target,
        "label": _graphml_edge_label(source, target, weight),
        "weight": weight,
        "abs_weight": abs_weight,
        "polarity": polarity,
        "contribution_by_mode": json.dumps(
            [dict(entry) for entry in edge.get("top_modes") or ()],
            sort_keys=True,
        ),
        "threshold_flag": True,
        "topN_flag": True,
        "intra_flag": bool(edge.get("is_intra", False)),
        "weight_ratio": weight_ratio,
        "edge_width": edge_width,
        "edge_color": GRAPHML_EDGE_COLORS.get(
            polarity, GRAPHML_EDGE_COLORS[0]
        ),
    }

    if z_contribution is not None:
        matrix_weight = float(z_contribution)
        if not math.isfinite(matrix_weight):
            raise MotifValidationError(
                f"GraphML edge {source!r}->{target!r} has a non-finite z_contribution"
            )
        metadata["z_contribution"] = matrix_weight

    return metadata
```

Notes:

- `weight` remains a Python `float`, so NetworkX emits GraphML `double`.
- `polarity` remains numeric.
- `z_contribution` remains numeric and is omitted when unavailable.
- `contribution_by_mode` remains a JSON string.
- `label` is a simple GraphML string.
- `edge_color` is a GraphML string for inspection/filtering.
- The nested yFiles style is inserted separately after NetworkX serialization.

If strict backward API compatibility for direct callers of `graphml_edge_metadata` is required, the new `abs_max` argument is optional, so existing calls such as:

```python
graphml_edge_metadata(edge)
```

remain valid.

### A8. Add the yFiles style injection helper

Add after `graphml_edge_metadata` and before `write_graphml_pathway`:

```python
def _inject_yfiles_edge_styles(
    body: str,
    graph: Any,
) -> str:
    """Augment NetworkX GraphML with deterministic yFiles edge visual properties."""
    graphml_namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("y", YFILES_GRAPHML_NAMESPACE)

    def graphml_tag(name: str) -> str:
        return f"{{{graphml_namespace}}}{name}"

    def yfiles_tag(name: str) -> str:
        return f"{{{YFILES_GRAPHML_NAMESPACE}}}{name}"

    root = ET.fromstring(body)
    graph_element = root.find(graphml_tag("graph"))
    if graph_element is None:
        raise MotifValidationError("generated GraphML has no <graph> element")

    style_key = ET.Element(
        graphml_tag("key"),
        {
            "id": GRAPHML_EDGE_STYLE_KEY,
            "for": "edge",
            "yfiles.type": "edgegraphics",
            "attr.name": "description",
            "attr.type": "string",
        },
    )
    graph_position = list(root).index(graph_element)
    root.insert(graph_position, style_key)

    keys_by_id = {
        element.get("id"): element
        for element in root.findall(graphml_tag("key"))
    }

    for edge_element in graph_element.findall(graphml_tag("edge")):
        source = edge_element.get("source")
        target = edge_element.get("target")
        if source is None or target is None:
            raise MotifValidationError(
                "generated GraphML contains an edge without source or target"
            )

        label = graph.edges[source, target].get("label")
        if not isinstance(label, str) or not label.strip() or label == "None":
            raise MotifValidationError(
                f"GraphML edge {source!r}->{target!r} has no usable label"
            )

        style_data = ET.SubElement(
            edge_element,
            graphml_tag("data"),
            {"key": GRAPHML_EDGE_STYLE_KEY},
        )
        edge_style = ET.SubElement(
            style_data,
            yfiles_tag("PolyLineEdge"),
        )
        ET.SubElement(
            edge_style,
            yfiles_tag("EdgeLabel"),
            {
                "alignment": "center",
                "autoSizePolicy": "content",
                "fontFamily": "Dialog",
                "fontSize": "12",
                "hasBackgroundColor": "false",
                "hasLineColor": "false",
                "horizontalTextPosition": "center",
                "verticalTextPosition": "center",
                "visible": "true",
            },
        ).text = "$label"
        ET.SubElement(
            edge_style,
            yfiles_tag("BorderStyle"),
            {
                "color": graph.edges[source, target]["edge_color"],
                "type": "line",
                "width": f"{graph.edges[source, target]['edge_width']:.3f}",
            },
        )
        ET.SubElement(
            edge_style,
            yfiles_tag("ArrowStyle"),
            {
                "type": "arrow",
                "visible": "true",
            },
        )

    del keys_by_id
    return ET.tostring(root, encoding="unicode")
```

Two cleanup improvements over the literal snippet above:

1. Remove the unused `keys_by_id` variable entirely.
2. The final helper should therefore end with:

```python
    return ET.tostring(root, encoding="unicode")
```

A cleaner complete helper is:

```python
def _inject_yfiles_edge_styles(body: str, graph: Any) -> str:
    """Augment NetworkX GraphML with deterministic yFiles edge visual properties."""
    graphml_namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("y", YFILES_GRAPHML_NAMESPACE)

    def graphml_tag(name: str) -> str:
        return f"{{{graphml_namespace}}}{name}"

    def yfiles_tag(name: str) -> str:
        return f"{{{YFILES_GRAPHML_NAMESPACE}}}{name}"

    root = ET.fromstring(body)
    graph_element = root.find(graphml_tag("graph"))
    if graph_element is None:
        raise MotifValidationError("generated GraphML has no <graph> element")

    style_key = ET.Element(
        graphml_tag("key"),
        {
            "id": GRAPHML_EDGE_STYLE_KEY,
            "for": "edge",
            "yfiles.type": "edgegraphics",
            "attr.name": "description",
            "attr.type": "string",
        },
    )
    root.insert(list(root).index(graph_element), style_key)

    for edge_element in graph_element.findall(graphml_tag("edge")):
        source = edge_element.get("source")
        target = edge_element.get("target")
        if source is None or target is None:
            raise MotifValidationError(
                "generated GraphML contains an edge without source or target"
            )

        attributes = graph.edges[source, target]
        label = attributes.get("label")
        if not isinstance(label, str) or not label.strip() or label == "None":
            raise MotifValidationError(
                f"GraphML edge {source!r}->{target!r} has no usable label"
            )

        style_data = ET.SubElement(
            edge_element,
            graphml_tag("data"),
            {"key": GRAPHML_EDGE_STYLE_KEY},
        )
        edge_style = ET.SubElement(style_data, yfiles_tag("PolyLineEdge"))
        ET.SubElement(
            edge_style,
            yfiles_tag("EdgeLabel"),
            {
                "alignment": "center",
                "autoSizePolicy": "content",
                "fontFamily": "Dialog",
                "fontSize": "12",
                "hasBackgroundColor": "false",
                "hasLineColor": "false",
                "horizontalTextPosition": "center",
                "verticalTextPosition": "center",
                "visible": "true",
            },
        ).text = "$label"
        ET.SubElement(
            edge_style,
            yfiles_tag("BorderStyle"),
            {
                "color": attributes["edge_color"],
                "type": "line",
                "width": f"{attributes['edge_width']:.3f}",
            },
        )
        ET.SubElement(
            edge_style,
            yfiles_tag("ArrowStyle"),
            {
                "type": "arrow",
                "visible": "true",
            },
        )

    return ET.tostring(root, encoding="unicode")
```

### Why `$label` is used in `<y:EdgeLabel>`

The edge will have two complementary representations:

```xml
<data key="d1">G04 → G01: -2.3453</data>
```

and:

```xml
<data key="yfiles_edge_graphics">
  <y:PolyLineEdge>
    <y:EdgeLabel ...>$label</y:EdgeLabel>
    ...
  </y:PolyLineEdge>
</data>
```

The first is ordinary GraphML metadata. The second tells yFiles/yEd to render the imported `label` model property. Relying only on the first `<data>` field is less reliable for automatic display because generic GraphML does not define visual semantics.

### A9. Pass `abs_max` when constructing edges

In `write_graphml_pathway`, after extracting `payload` and before the graph construction, calculate:

```python
    abs_max_raw = payload.get("abs_max")
    abs_max = float(abs_max_raw) if abs_max_raw is not None else 0.0
    if not math.isfinite(abs_max) or abs_max < 0.0:
        raise MotifValidationError("pathway GraphML abs_max must be finite and non-negative")
```

Then update edge creation around current lines 3589–3596:

```python
    for edge in edges:
        source = _required_graphml_text(
            edge.get("source"),
            field="source",
            context="pathway edge",
        )
        target = _required_graphml_text(
            edge.get("target"),
            field="target",
            context=f"pathway edge {source!r}->{target!r}",
        )
        metadata = graphml_edge_metadata(
            edge,
            abs_max=abs_max,
            z_contribution=(z_by_edge or {}).get((source, target)),
        )
        graph.add_edge(source, target, **metadata)
```

### A10. Add graph metadata

Update `graph.graph.update()` around current lines 3599–3623 to:

```python
    graph.graph.update(
        {
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "abs_max": abs_max,
            "weight_normalization": PATHWAY_WEIGHT_NORMALIZATION,
            "edge_style": json.dumps(
                {
                    "width_formula": (
                        "1.0 + 4.0 * "
                        "min(1.0, abs_weight / abs_max)"
                    ),
                    "width_min": GRAPHML_EDGE_WIDTH_MIN,
                    "width_max": GRAPHML_EDGE_WIDTH_MAX,
                    "color_by_polarity": {
                        str(polarity): color
                        for polarity, color in GRAPHML_EDGE_COLORS.items()
                    },
                },
                sort_keys=True,
            ),
            "pathway_source": str(
                source
                if source is not None
                else payload.get("source") or DEFAULT_PATHWAY_SOURCE
            ),
            "pathway_weight_rule": str(
                config.pathway_weight
                if config is not None
                else payload.get("weight") or DEFAULT_PATHWAY_WEIGHT
            ),
            "threshold": float(
                config.pathway_edge_threshold
                if config is not None
                else (
                    edge_threshold
                    if edge_threshold is not None
                    else DEFAULT_PATHWAY_EDGE_THRESHOLD
                )
            ),
            "topN": int(
                config.pathway_top_edges
                if config is not None
                else (
                    top_edges
                    if top_edges is not None
                    else DEFAULT_PATHWAY_TOP_EDGES
                )
            ),
        }
    )
```

### A11. Keep the existing key validation

No algorithm change is needed to the current checks around lines 3625–3639. Because the new fields are included in `GRAPHML_EDGE_KEYS`, the existing logic accepts them while continuing to reject unknown fields:

```python
    for node_id, attributes in graph.nodes(data=True):
        unknown = sorted(set(attributes) - set(GRAPHML_NODE_KEYS))
        if unknown:
            raise MotifValidationError(
                f"GraphML node {node_id} carries unknown keys {unknown}"
            )

    for source_id, target_id, attributes in graph.edges(data=True):
        unknown = sorted(set(attributes) - set(GRAPHML_EDGE_KEYS))
        if unknown:
            raise MotifValidationError(
                f"GraphML edge {source_id}->{target_id} "
                f"carries unknown keys {unknown}"
            )

    unknown_graph = sorted(set(graph.graph) - set(GRAPHML_GRAPH_KEYS))
    if unknown_graph:
        raise MotifValidationError(
            f"the GraphML graph metadata carries unknown keys {unknown_graph}"
        )
```

The yFiles style key is not a NetworkX edge attribute, so it does not enter this validation. It is added only during XML post-processing and has its own fixed, controlled key ID.

### A12. Inject yFiles styles after NetworkX generation

Replace current lines 3641–3645 with:

```python
    try:
        body = "\n".join(nx.generate_graphml(graph))
        body = _inject_yfiles_edge_styles(body, graph)
    except MotifValidationError:
        raise
    except Exception as exc:
        raise MotifValidationError(
            f"could not serialise the GraphML pathway: {exc}"
        ) from exc

    return _write_text_atomic(
        path,
        f"{XML_DECLARATION}\n{body}\n",
    )
```

This retains atomic writing. A failure during style injection occurs before `_write_text_atomic`, so no partial artifact is created.

### A13. Update the module-level GraphML documentation

The module docstring around lines 8–10 should be updated to make the visual behavior explicit:

```python
as a NetworkX GraphML document for interactive exploration in yEd.  Edges carry
scalar metadata, human-readable labels, and deterministic yFiles visual
properties that map absolute weight to line width and signed polarity to color.
```

Do not alter any pipeline, schema, or config documentation.

---

## B. `/home/mr-miracle/projects/eigen-decomposition/src/clustering/__init__.py`

### B1. No new public helper is required

The implementation helpers should be private:

```python
_required_graphml_text
_graphml_edge_label
_graphml_edge_width
_inject_yfiles_edge_styles
```

Therefore, do not add them to `TYPE_CHECKING` or `__all__`.

The existing public exports are already correct:

```python
"GRAPHML_EDGE_KEYS",
"GRAPHML_GRAPH_KEYS",
"GRAPHML_NODE_KEYS",
"graphml_edge_metadata",
"graphml_node_metadata",
"write_graphml_pathway",
```

No export entries need to be added merely because their contents change.

### B2. Update the package overview

Extend the public-surface bullet around line 22:

```python
* :func:`write_artifact_set` -- persist the requested artifact set.
* :func:`write_graphml_pathway` -- write yEd-facing GraphML with readable edge
  labels and deterministic yFiles weight/polarity styling.
* :func:`main` -- the command line interface.
```

### B3. Optional constant exports

If the project convention requires styling constants to be public, export:

```python
GRAPHML_EDGE_COLORS
GRAPHML_EDGE_WIDTH_MAX
GRAPHML_EDGE_WIDTH_MIN
GRAPHML_EDGE_STYLE_KEY
YFILES_GRAPHML_NAMESPACE
```

in three places:

1. `TYPE_CHECKING` import list;
2. `__all__`;
3. No runtime implementation is needed because the existing lazy `__getattr__` resolves them.

This is optional. The lower-churn recommendation is to keep them private/non-exported and leave the public API unchanged.

---

# 3. Exact resulting yFiles XML

For an edge such as:

```text
G04 → G01, weight = -2.3452997
```

the ordinary NetworkX data should include:

```xml
<key id="d..." for="edge" attr.name="label" attr.type="string"/>
<key id="d..." for="edge" attr.name="weight" attr.type="double"/>
<key id="d..." for="edge" attr.name="abs_weight" attr.type="double"/>
<key id="d..." for="edge" attr.name="polarity" attr.type="long"/>
<key id="d..." for="edge" attr.name="contribution_by_mode" attr.type="string"/>
<key id="d..." for="edge" attr.name="weight_ratio" attr.type="double"/>
<key id="d..." for="edge" attr.name="edge_width" attr.type="double"/>
<key id="d..." for="edge" attr.name="edge_color" attr.type="string"/>
```

The edge should then contain the yFiles style:

```xml
<data key="yfiles_edge_graphics">
  <y:PolyLineEdge>
    <y:EdgeLabel
        alignment="center"
        autoSizePolicy="content"
        fontFamily="Dialog"
        fontSize="12"
        hasBackgroundColor="false"
        hasLineColor="false"
        horizontalTextPosition="center"
        verticalTextPosition="center"
        visible="true">$label</y:EdgeLabel>
    <y:BorderStyle
        color="#C62828"
        type="line"
        width="5.000"/>
    <y:ArrowStyle
        type="arrow"
        visible="true"/>
  </y:PolyLineEdge>
</data>
```

For weaker edges, the same style has a smaller `BorderStyle/@width`. Positive edges use blue:

```xml
<y:BorderStyle color="#1565C0" type="line" width="..."/>
```

Zero-polarity edges use gray:

```xml
<y:BorderStyle color="#757575" type="line" width="..."/>
```

The GraphML root should declare:

```xml
xmlns:y="http://www.yworks.com/xml/graphml"
```

and the visual key should declare:

```xml
<key
    id="yfiles_edge_graphics"
    for="edge"
    yfiles.type="edgegraphics"
    attr.name="description"
    attr.type="string"/>
```

---

# 4. Metadata completeness

## Preserve these types

| Field | Required GraphML type |
|---|---|
| `label` | string |
| `weight` | double |
| `abs_weight` | double |
| `polarity` | integer/long |
| `z_contribution` | double, optional |
| `contribution_by_mode` | string containing JSON |
| `weight_ratio` | double |
| `edge_width` | double |
| `edge_color` | string |
| `threshold_flag` | boolean |
| `topN_flag` | boolean |
| `intra_flag` | boolean |

The helper should not do this:

```python
"weight": json.dumps(weight)
```

or:

```python
"weight": str(weight)
```

Both would unnecessarily turn numerical metadata into text.

## Missing fields added by this patch

### `label`

Makes direct edge inspection and label mapping possible.

### `weight_ratio`

This is preferable to styling from raw `abs_weight`, because raw scale changes with source, configuration, and dataset.

### `edge_width`

Stores the exact width used in the yFiles block, making the visual result inspectable and testable.

### `edge_color`

Stores the exact color used in the yFiles block.

### `abs_max` at graph level

Records the denominator for the width normalization.

### `edge_style` at graph level

Records the mapping formula and color legend. It should be a deterministic sorted-key JSON string.

---

# 5. Updated test cases

Tests are outside the two production-file scope, but they are required deliverables.

## A. Metadata helper test

In `/home/mr-miracle/projects/eigen-decomposition/tests/test_functional_motifs.py`, extend the existing `test_04d_graphml_metadata_helpers` around current lines 1580–1611.

Add these assertions after constructing `plain`:

```python
    assert plain["label"] == "G01 → G02: -1.5000"
    assert plain["weight_ratio"] == 1.0
    assert plain["edge_width"] == 5.0
    assert plain["edge_color"] == "#C62828"
```

Add a positive-edge case:

```python
    positive = graphml_edge_metadata(
        {
            **edge,
            "weight": 0.5,
            "abs_weight": 0.5,
            "polarity": 1,
        },
        abs_max=1.0,
    )
    assert positive["label"] == "G01 → G02: +0.5000"
    assert positive["weight_ratio"] == 0.5
    assert positive["edge_width"] == 3.0
    assert positive["edge_color"] == "#1565C0"
```

Add a neutral-edge case:

```python
    neutral = graphml_edge_metadata(
        {
            **edge,
            "weight": 0.0,
            "abs_weight": 0.0,
            "polarity": 0,
        },
        abs_max=0.0,
    )
    assert neutral["label"] == "G01 → G02: 0.0000"
    assert neutral["weight_ratio"] == 0.0
    assert neutral["edge_width"] == 1.0
    assert neutral["edge_color"] == "#757575"
```

Retain:

```python
    assert "z_contribution" not in plain
    assert graphml_edge_metadata(
        edge, z_contribution=0.42
    )["z_contribution"] == 0.42
```

Add numeric-type checks:

```python
    for key in ("weight", "abs_weight", "z_contribution"):
        assert isinstance(plain.get(key, 0.42), float)
    assert isinstance(plain["polarity"], int)
    assert isinstance(json.loads(plain["contribution_by_mode"]), list)
```

## B. Node-label validation tests

Add a dedicated test:

```python
@pytest.mark.parametrize(
    "node",
    [
        {"node_id": "G01"},
        {"node_id": "G01", "group_id": "", "label": "G01"},
        {"node_id": "G01", "group_id": "G01", "label": ""},
        {"node_id": "G01", "group_id": "G01", "label": "None"},
    ],
)
def test_04d_graphml_rejects_invalid_node_identity(node) -> None:
    with pytest.raises(MotifValidationError, match="graph_id|label"):
        graphml_node_metadata(node)
```

Add a fallback test:

```python
def test_04d_graphml_node_label_falls_back_to_group_id() -> None:
    metadata = graphml_node_metadata(
        {
            "node_id": "G01",
            "group_id": "G01",
            "label": "",
            "size": 3,
        }
    )
    assert metadata["label"] == "G01"
```

The exact regex may need to account for the chosen contextual error wording.

## C. GraphML scalar round-trip test

In the existing `test_04d_graphml_round_trips`, add for every node:

```python
        assert isinstance(attributes["label"], str)
        assert attributes["label"].strip()
        assert attributes["label"] != "None"
        assert attributes["label"] == node_id
```

Add for every edge:

```python
        assert isinstance(attributes["label"], str)
        assert attributes["label"] == f"{source} → {target}: {attributes['weight']:+.4f}"
        assert attributes["weight_ratio"] <= 1.0
        assert attributes["edge_width"] == pytest.approx(
            1.0 + 4.0 * attributes["weight_ratio"]
        )
        assert attributes["edge_color"] in {
            "#C62828",
            "#1565C0",
            "#757575",
        }
```

For the synthetic dataset with nonzero signed values, also assert:

```python
        assert attributes["edge_color"] == (
            "#C62828" if attributes["weight"] < 0 else "#1565C0"
        )
```

Add graph metadata checks:

```python
    assert graph.graph["abs_max"] == analysis.pathway["abs_max"]
    edge_style = json.loads(graph.graph["edge_style"])
    assert edge_style["width_min"] == 1.0
    assert edge_style["width_max"] == 5.0
    assert edge_style["color_by_polarity"] == {
        "-1": "#C62828",
        "0": "#757575",
        "1": "#1565C0",
    }
```

## D. yFiles XML structure test

Add an XML parsing test:

```python
def test_04d_graphml_contains_yfiles_edge_styles(
    synthetic_spectrum, tmp_path: Path
) -> None:
    from xml.etree import ElementTree as ET

    analysis = _pathway_inputs(synthetic_spectrum)
    paths = artifact_paths(analysis, tmp_path)
    written = write_graphml_pathway(
        paths["graphml"],
        analysis.pathway,
        groups=analysis.groups,
        config=analysis.config,
    )

    graphml_ns = "http://graphml.graphdrawing.org/xmlns"
    yfiles_ns = "http://www.yworks.com/xml/graphml"
    root = ET.parse(written).getroot()

    style_key = next(
        key
        for key in root.findall(f"{{{graphml_ns}}}key")
        if key.get("yfiles.type") == "edgegraphics"
    )
    assert style_key.get("id") == "yfiles_edge_graphics"
    assert style_key.get("for") == "edge"
    assert style_key.get("attr.type") == "string"

    edges = root.findall(f".//{{{graphml_ns}}}edge")
    assert edges
    for edge_element in edges:
        style_data = next(
            data
            for data in edge_element.findall(f"{{{graphml_ns}}}data")
            if data.get("key") == style_key.get("id")
        )
        polyline = style_data.find(f"{{{yfiles_ns}}}PolyLineEdge")
        assert polyline is not None

        edge_label = polyline.find(f"{{{yfiles_ns}}}EdgeLabel")
        assert edge_label is not None
        assert edge_label.text == "$label"

        border = polyline.find(f"{{{yfiles_ns}}}BorderStyle")
        assert border is not None
        assert border.get("color") in {"#C62828", "#1565C0", "#757575"}
        assert 1.0 <= float(border.get("width")) <= 5.0

        arrow = polyline.find(f"{{{yfiles_ns}}}ArrowStyle")
        assert arrow is not None
        assert arrow.get("type") == "arrow"
```

## E. Width/color consistency test

The XML width and color should match the ordinary edge metadata:

```python
def test_04d_graphml_style_matches_edge_metadata(
    synthetic_spectrum, tmp_path: Path
) -> None:
    nx = pytest.importorskip("networkx")
    from xml.etree import ElementTree as ET

    analysis = _pathway_inputs(synthetic_spectrum)
    paths = artifact_paths(analysis, tmp_path)
    written = write_graphml_pathway(
        paths["graphml"],
        analysis.pathway,
        config=analysis.config,
    )

    graph = nx.read_graphml(written)
    graphml_ns = "http://graphml.graphdrawing.org/xmlns"
    yfiles_ns = "http://www.yworks.com/xml/graphml"
    root = ET.parse(written).getroot()

    for element in root.findall(f".//{{{graphml_ns}}}edge"):
        source = element.get("source")
        target = element.get("target")
        border = element.find(
            f".//{{{yfiles_ns}}}BorderStyle"
        )
        metadata = graph.edges[source, target]

        assert border.get("color") == metadata["edge_color"]
        assert float(border.get("width")) == pytest.approx(
            metadata["edge_width"]
        )
```

## F. Unknown-key test

Retain the existing `test_04d_graphml_rejects_unknown_keys`, because it verifies the whitelist still catches helper regressions.

Add a second check for graph metadata:

```python
def test_04d_graphml_rejects_unknown_graph_key(
    synthetic_spectrum, tmp_path: Path, monkeypatch
) -> None:
    import src.clustering.functional_motifs as fm

    analysis = _pathway_inputs(synthetic_spectrum)
    original_generate_graphml = fm.networkx.generate_graphml if hasattr(
        fm, "networkx"
    ) else None
```

Because NetworkX is intentionally imported lazily, do not access `fm.networkx`. Instead, the existing helper-level unknown-field test is sufficient. A cleaner approach is to test a private validation helper if one is extracted, or monkeypatch the metadata helper as the current test already does.

## G. Determinism test

Keep:

```python
    assert first.read_bytes() == second.read_bytes()
```

`ElementTree` serialization is deterministic for the same input, but this test must remain because the implementation now performs XML post-processing.

## H. JSON/NPZ/config regression test

Add or extend a test that hashes/serializes the analysis with and without GraphML writing:

```python
def test_04d_graphml_fields_do_not_change_json_npz_or_config_hash(
    synthetic_spectrum, tmp_path: Path
) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)

    before_json = json.dumps(
        analysis.payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    before_arrays = {
        key: value.copy()
        for key, value in analysis.arrays.items()
        if isinstance(value, np.ndarray)
    }
    before_hash = analysis.config.config_hash

    write_graphml_pathway(
        tmp_path / "motifs.pathway.graphml",
        analysis.pathway,
        config=analysis.config,
    )

    after_json = json.dumps(
        analysis.payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    after_arrays = {
        key: value
        for key, value in analysis.arrays.items()
        if isinstance(value, np.ndarray)
    }
    after_hash = analysis.config.config_hash

    assert after_json == before_json
    assert after_hash == before_hash
    assert set(after_arrays) == set(before_arrays)
    for key in before_arrays:
        np.testing.assert_array_equal(after_arrays[key], before_arrays[key])
```

If `analysis` does not expose `config_hash` directly, compare:

```python
analysis.config.config_hash
```

against the unchanged `MOTIF_CONFIG_FIELDS` result, or compare the JSON/config before and after writing.

---

# 6. Schema-level test updates

In `/home/mr-miracle/projects/eigen-decomposition/tests/test_motif_schema.py`, around current lines 307–319, add:

```python
    for node in pathway["nodes"]:
        attributes = graph.nodes[node["node_id"]]
        assert set(attributes) <= set(GRAPHML_NODE_KEYS)
        assert attributes["group_id"] == node["group_id"]
        assert attributes["label"] == node["label"]
        assert attributes["label"]
        assert attributes["label"] != "None"
        assert attributes["size"] == node["size"]
        assert json.loads(attributes["region_composition"]) == node["region_composition"]

    abs_max = pathway["abs_max"]
    for edge in pathway["edges"]:
        attributes = graph.edges[edge["source"], edge["target"]]
        assert set(attributes) <= set(GRAPHML_EDGE_KEYS)
        assert attributes["label"] == (
            f"{edge['source']} → {edge['target']}: "
            f"{edge['weight']:+.4f}"
        )
        assert attributes["weight"] == edge["weight"]
        assert attributes["abs_weight"] == edge["abs_weight"]
        assert attributes["polarity"] == edge["polarity"]
        assert attributes["intra_flag"] == edge["is_intra"]
        assert json.loads(attributes["contribution_by_mode"]) == edge["top_modes"]

        expected_ratio = (
            min(1.0, edge["abs_weight"] / abs_max)
            if abs_max > 0.0
            else 0.0
        )
        assert attributes["weight_ratio"] == pytest.approx(expected_ratio)
        assert attributes["edge_width"] == pytest.approx(
            1.0 + 4.0 * expected_ratio
        )
```

Add:

```python
    assert graph.graph["abs_max"] == pathway["abs_max"]
    assert set(graph.graph) - {"node_default", "edge_default"} <= set(
        GRAPHML_GRAPH_KEYS
    )
```

This test is useful because it proves the GraphML-only fields do not become pathway-payload fields.

---

# 7. Updated documentation sections

## A. README GraphML section

Add a subsection with this content:

### GraphML pathway artifact

`motifs.pathway.graphml` is a directed NetworkX GraphML graph intended for yEd.

Nodes include:

- `group_id`
- `label`
- group size and coherence
- dominant mode
- region composition
- background/singleton flags
- optional participation centroid

Edges include:

- `label`, formatted as `source → target: signed weight`
- numeric `weight`, `abs_weight`, and `polarity`
- optional numeric `z_contribution`
- JSON-encoded `contribution_by_mode`
- `weight_ratio`
- computed `edge_width`
- computed `edge_color`
- threshold/top-N/intra-group flags

Each edge also contains a yFiles `edgegraphics` visual-properties block:

- edge width ranges from `1.0` to `5.0` according to  
  `abs_weight / abs_max`;
- negative edges are red (`#C62828`);
- positive edges are blue (`#1565C0`);
- zero-polarity edges are gray (`#757575`);
- edge labels are rendered through the yFiles `EdgeLabel` style.

Graph-level metadata records the exact normalization formula and palette under `edge_style`.

GraphML changes do not alter the JSON pathway schema, NPZ arrays, configuration hash, or other Phase 01–04 artifacts.

## B. `execution-plans/04D_pathways.md`

Extend the edge key table with:

```text
- `label`
- `weight_ratio`
- `edge_width`
- `edge_color`
```

Add:

```text
The GraphML backend remains NetworkX. NetworkX generates the standard
GraphML document and scalar data fields; the writer then augments that
serialized document with deterministic yFiles/yEd `edgegraphics` style blocks.
No config field or CLI option is introduced.
```

## C. `execution-plans/04D_pathways_detailed.md`

Update §7 with:

- ordinary GraphML `label` data field;
- yFiles namespace and style key;
- width formula;
- polarity palette;
- optional `z_contribution`;
- deterministic XML post-processing;
- NetworkX round-trip compatibility;
- distinction between ordinary metadata and yFiles visual properties.

## D. Compatibility statement

Add:

```text
The visual enhancement is confined to GraphML serialization. `pathway.nodes`,
`pathway.edges`, frozen JSON key sets, NPZ arrays, config fields, and
`config_hash` are unchanged. Existing `graphml_edge_metadata(edge)` calls remain
valid because the new `abs_max` argument is optional.
```

---

# 8. Backward-compatibility safeguards

## JSON

No changes should occur to:

- `PathwayNode.to_dict()`
- `PathwayEdge.to_dict()`
- `PathwayDiagram.to_dict()`
- `analysis.pathway`
- `analysis.metadata`
- frozen schema constants
- `validate_motif_payload_schema`

The new `label`, `weight_ratio`, `edge_width`, and `edge_color` values must exist only on the temporary NetworkX graph.

## NPZ

Do not add any array.

The NPZ must remain at the Phase 04D count—24 arrays when pathway groups exist and 19 in the grouping-none case.

## Configuration

Do not add anything to:

```python
MOTIF_CONFIG_FIELDS
```

or `MotifConfig`.

The fixed visualization constants must not be added to the configuration hash. Therefore:

```python
config_hash
```

must remain byte-for-byte unchanged for the same input and CLI configuration.

## Existing helper API

The change to:

```python
graphml_edge_metadata(edge, *, z_contribution=None)
```

should become:

```python
graphml_edge_metadata(
    edge,
    *,
    abs_max=0.0,
    z_contribution=None,
)
```

because `abs_max` is keyword-only and optional. Existing direct calls remain valid.

## Phases 01–03

There should be no edits to:

- parsing
- matrix construction
- spectral decomposition
- shared IO
- configuration fields
- Phase 01–03 tests or artifacts

---

# 9. Implementation order

1. Add `ElementTree` import and private style constants.
2. Extend all three GraphML key sets.
3. Add `_required_graphml_text`.
4. Harden `graphml_node_metadata`.
5. Add edge-label and width helpers.
6. Extend `graphml_edge_metadata`.
7. Add `_inject_yfiles_edge_styles`.
8. Pass `abs_max` and `z_by_edge` into edge metadata.
9. Add graph-level `abs_max` and `edge_style`.
10. Call style injection after NetworkX serialization.
11. Update package documentation in `__init__.py`.
12. Update metadata-helper, round-trip, determinism, style, and unknown-key tests.
13. Update schema-level assertions.
14. Update README and Phase 04D plan documentation.
15. Run focused tests, then the full suite.
16. Generate and inspect a real GraphML file.
17. If practical, import the generated file into the installed yEd version and verify:
    - `G04 → G01: -2.3453` is rendered;
    - the strongest edge has the greatest width;
    - negative edges are red;
    - arrows follow source-to-target direction;
    - NetworkX round-trip still exposes all ordinary scalar attributes.

---

# 10. Final acceptance criteria

The patch is complete when:

- Every node has a non-empty label that cannot be the literal `"None"`.
- Every edge has a simple string label such as `G04 → G01: -2.3453`.
- Edge `weight`, `abs_weight`, `polarity`, and available `z_contribution` are numeric.
- `contribution_by_mode` remains JSON.
- Every edge has one yFiles `edgegraphics` data block.
- Edge labels use a yFiles `EdgeLabel` referencing `$label`.
- Edge width is in `[1.0, 5.0]` and increases with `abs_weight / abs_max`.
- Negative edges are red, positive edges blue, and zero-polarity edges gray.
- All ordinary edge attributes belong to `GRAPHML_EDGE_KEYS`.
- Unknown ordinary keys are still rejected.
- The yFiles namespace and key ID are deterministic.
- Two writes with the same input remain byte-identical.
- `networkx.read_graphml()` successfully round-trips ordinary graph metadata.
- `motifs.json`, `motifs.data.json`, and `motifs.npz` remain unchanged.
- `config_hash` remains unchanged.
- No Phase 01–03 production behavior is modified.