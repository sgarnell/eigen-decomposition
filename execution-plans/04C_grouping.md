 # Sub-phase 04C — Functional neuron groups
 

**Input/Output.** `groups` populated in `motifs.json` (a **total partition**, including `G00` background); `group_labels` `(N,)`, `group_centroids` `(G,k)`, `group_sizes` `(G,)` added to `motifs.npz`; group block of `metadata` and `motifs.data.json` populated.

**New symbols.** `assign_groups`, `initial_group_labels`, `merge_similar_groups`, `split_large_groups`, `loadings_clustering`, `build_groups`, `group_from_labels`, `group_centroid`, `group_coherence`, `region_composition`.

**CLI added.** `--grouping {motif,loadings,hybrid,none}` (default `motif` — **the locked decision**), `--polarity-split`, `--polarity-min-members` (3), `--max-group-size` (12), `--n-groups` (`auto`), `--linkage` (`average`), `--affinity` (`cosine`), `--merge-threshold` (0.9).

**Validation rules added.** grouping config enums/ranges (error); `partition` invariant violation (error); background-present / singleton-present (info); **sklearn zero-vector guard** (error → removed before clustering, re-attached as `G00`).

**Reference numbers (this sub-phase owns them, default `motif` grouping).**

| Quantity | Value |
|---|---|
| `n_groups` | **22** = `G00` background + `G01…G21` ↔ `M01…M21` |
| non-background group sizes | 10, 10, 9, 7, 7, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 2, 1, 1, 1 |
| `largest_group` / `smallest_group` / `mean_group_size` | 10 / 1 / 5.136364 |
| `n_background_groups` / `n_singleton_groups` | 1 / 3 |
| `n_polarity_split_groups` (default, split off) | 0 |
| group `coherence` min / max / mean | −0.155531 / 0.876450 / 0.147827 |
| qualitative check | `G01 ≈ FB4K/hDeltaA`, `G02 ≈ FB4E/hDeltaI`, `G03 ≈ FB5O/R/S/U/hDeltaH` |

**Tests (≈30 new + 2 slow).** `argmax` assignment + ties; partition invariant (`Σ sizes == N`, disjoint, sorted, `group_labels` consistency); each non-background member's `dominant_mode` is its `argmax`; `polarity-split` with the minority guard (synthetic mode with both signs); `loadings` with `--n-groups` int and `auto`; hybrid merge on synthetic groups with known centroid cosine; `none`; the zero-vector guard (asserted both ways); region composition/`cent_mean`; determinism; slow tests for the 22 groups/sizes/coherence.

**Acceptance.** 04C numbers reproduced; the partition invariant holds for all four grouping modes; 04A/04B tests unchanged and green.

**Non-goals.** Pathway, plots, k-means/spectral/HDBSCAN/UMAP (documented future).

**Deliverables.** `execution-plans/04C_grouping.md`; 04C scope added to the module + `__init__.py`; new tests; README grouping subsection (including the documented empirical finding that intra-group loading coherence is low, so `loadings` grouping needs an explicit `--n-groups`).
