# Sub-phase 04C — Functional neuron groups: detailed design

*Implementation design for `execution-plans/04C_grouping.md`; 04A/04B behavior remains additive and unchanged.*

## 1. Scope and contract

04C populates the frozen `groups` payload block as a total partition of all neurons, adds
`group_labels (N,)`, `group_centroids (G,k)`, and `group_sizes (G,)` to the NPZ sidecar, and
fills the eight existing group metadata fields. It adds no schema keys, config fields, or
dependencies beyond the already-declared scikit-learn environment dependency. 04D consumes the
result as pathway nodes; pathway construction and plotting are out of scope.

The default is the locked `--grouping motif` mode. It is NumPy-only and deterministic. The other
modes are `loadings`, `hybrid`, and `none`; the latter preserves the neutral groups block and
does not emit placeholder group arrays.

## 2. Definitions

### 2.1 Motif assignment

For participation matrix `P (N,k)`, `assign_groups` assigns each nonzero row to
`1 + argmax(P[i])`; NumPy's first-index tie behavior is the required lowest-mode tie break. A row
whose complete participation vector is zero receives provisional label `0` and becomes the
background group. This is the same dominant-mode rule already used by `save_data_payload`.

### 2.2 Stable group ids

`group_from_labels` keeps label `0` as `G00`. Default motif labels are one-based mode numbers and
are preserved so `G01` ↔ `M01`; arbitrary loading-cluster labels are ordered by `(-size,
raw_label)` for deterministic ids. Members are always emitted in ascending neuron-index order.
The mode-aligned reference vector is `[1,6,10,7,7,7,10,9,2,7,1,1,5,1,4,5,5,4,6,7,4,4]`; its
non-background size multiset is the brief's `10,10,9,7,7,7,7,7,6,6,5,5,5,4,4,4,4,2,1,1,1`.

### 2.3 Signed participation and coherence

The dominant-axis polarity is exactly the 04A `extract_motif` rule: use the sign of `left` when
`abs(left) >= abs(right)`, otherwise the sign of `right`, falling through to the other axis and
then `+1` for two zeros. Define `S[i,m] = polarity(i,m) * P[i,m]`. For each group with at least
two members, coherence is the mean of the strict-upper-triangle pairwise cosine matrix of the
members' rows in `S`. Singleton/background coherence is `None`.

This definition reproduces the reference coherence range exactly: minimum `-0.155531`, maximum
`0.876450`, mean `0.147827` over the 18 non-singleton non-background groups. It also documents
the empirical finding that motif-group loading coherence is low.

### 2.4 Other grouping modes

* `loadings`: remove all-zero participation rows, build `[loadings_left | loadings_right]`, and
  run `sklearn.cluster.AgglomerativeClustering`. `metric` is the configured affinity and
  linkage is configured directly. Integer `n_groups` is used as `n_clusters`; `auto` resolves
  to `k`, rather than a distance cut, because a `0.1` cosine-distance cut produces 103 fragments
  on the reference data. Zero rows are reattached to G00.
* `hybrid`: begin with motif argmax labels, then apply similarity-based merging in `[L|R]`
  loading space. Merging is only performed while an oversized group exists and the highest
  centroid cosine is at least `merge_threshold`; ties are deterministic.
* `none`: `groups=[]`, all group metadata remains neutral, and no group arrays are added.

### 2.5 Polarity split and merge

When `--polarity-split` is enabled, split a group's members by the dominant polarity in its seed
mode only when both signs exist and the minority has at least `polarity_min_members` members.
`n_polarity_split_groups` counts created groups. The default is off, giving zero splits.

`merge_similar_groups` computes feature-space centroids and merges the most similar pair only
when a group exceeds `max_group_size`; the threshold is inclusive. Default motif grouping has
largest size 10, so the default reference run performs no merge.

## 3. Payload and metadata

The default reference output must contain 22 groups including G00, group sizes in id order
`[1, 6, 10, 7, 7, 7, 10, 9, 2, 7, 1, 1, 5, 1, 4, 5, 5, 4, 6, 7, 4, 4]`, largest 10,
smallest 1, mean `5.136364`, one background group, and three non-background singletons.

Each group contains the frozen `GROUP_KEYS`: id/label, sorted members and counts, seed dominant
mode (`None` for G00), sorted region composition, mean `cent` when available, coherence, and
background/singleton flags. `group_centroids` is the mean **unsigned participation** vector per
group (G00 is a zero row); merge/coherence feature vectors remain internal implementation data.
`motifs.data.json` forwards groups and adds the final integer `group_id` to each neuron row.

## 4. Validation and diagnostics

Add config errors for invalid polarity minimum, group-size cap, group count, merge threshold, and
the incompatible `ward`+`cosine` combination before sklearn is called. Add an error for any
partition violation (overlap, omitted neuron, unsorted members, inconsistent labels/sizes).
Log background, singleton, and zero-vector-guard conditions at info level. Use new codes
`CODE_PARTITION`, `CODE_BACKGROUND`, `CODE_SINGLETON`, and `CODE_ZERO_VECTOR`; the frozen schema
validator remains unchanged.

## 5. Implementation sequence

1. Add grouping helpers, constants, and lazy exports.
2. Implement deterministic motif assignment, renumbering, centroids, regions, coherence, and
   partition checks.
3. Add polarity splitting, similarity merging, loading clustering, zero-vector handling, and
   `none` dispatch.
4. Wire 04C into `build_motif_analysis`, metadata, save-data, sidecar verification, CLI, stats,
   and summary output; set stage to `04C`.
5. Add approximately 30 fast tests and two slow reference tests. Update only assertions that
   previously pinned neutral 04C values; preserve all 04A/04B tests and semantics otherwise.
6. Update README with CLI examples, NPZ fields, low-coherence finding, explicit `--n-groups`
   guidance, and the status row. Regenerate artifacts only after the full validation passes.

## 6. Tests and acceptance

Tests cover argmax and ties; total/disjoint/sorted partition and label consistency; dominant
mode; polarity minority guard; loading `auto` and integer counts; hybrid centroid merge; `none`;
zero-vector removal/re-attachment; region/cent statistics; determinism; schema and sidecar
round-trip. Slow tests assert the 22-group reference sizes, coherence values, background and
qualitative compositions. Acceptance requires all four grouping modes to satisfy the partition
invariant, 04A/04B reference values to remain unchanged, 21 sidecar arrays for active grouping,
and byte-identical outputs under a fixed `SOURCE_DATE_EPOCH`.