Please update the Phase 02 implementation (build_square_matrix.py) and the
execution-plans/02_build_square_matrix.md document to include the following
CLI options, behaviors, and outputs. These requirements must be implemented
cleanly, consistently, and without breaking Phase 01 or Phase 03 semantics.

=====================================================================
PHASE 02 — REQUIRED CLI OPTIONS
=====================================================================

A. --plot
    - Generate a 2D heatmap PNG of the unified-weight Z matrix.
    - Use a perceptually uniform colormap (viridis or plasma).
    - Save to <stem>/z_matrix.png.
    - ALSO display the heatmap in a GUI popup window using plt.show().
    - This is the ONLY popup window in Phase 02.

B. --symmetric
    - Compute Z_sym = (Z + Z.T) / 2.
    - Justification:
        Symmetric matrices allow true eigen-decomposition with real
        eigenvalues and orthogonal eigenvectors. This is required for
        Phase 03’s spectral pipeline when symmetric modes are desired.
    - When enabled, both JSON and NPZ must contain Z_sym instead of Z.
    - The save-data file must record whether symmetric mode was used.

C. --normalize
    - Apply row/column normalization or weight scaling.
    - Provide at least:
        --normalize rows
        --normalize cols
        --normalize unit   (divide by max absolute weight)
    - Record normalization mode in metadata and in the save-data file.

D. --config-hash
    - Ensure filenames include <config_hash8> for any non-default config.
    - Hash must incorporate: symmetric, normalize, filters, unified-weight
      parameters, and any other config flags.

E. --no-sidecar
    - Suppress writing z_matrix.npz.
    - JSON remains canonical.
    - load_z_matrix() must still work.

F. --strict
    - Fail on any validation issue.
    - Strict mode applies to both JSON and NPZ writing.

H. --save-data
    - Write a JSON file containing:
        * unified weights per pair (source, target, weight)
        * pre_z, post_z
        * symmetric flag
        * normalization mode
        * Z statistics (min, max, mean, std)
        * histogram bins of weights
        * sparsity (fraction of zeros)
        * positive/negative counts
        * row/column norms
        * spectral radius (if symmetric)
        * density and reciprocity (copied from Phase 01)
    - Filename: <stem>/z_matrix.data.json

I. --stats
    - Print summary statistics of Z to the terminal.
    - Include:
        * min, max, mean, std
        * sparsity
        * positive/negative counts
        * row/column norms
        * spectral radius (if symmetric)
    - After printing, show a terminal “popup-style” ASCII summary block.
    - This is NOT a GUI popup. It is terminal-only.

=====================================================================
PHASE 02 — REQUIRED OUTPUT ARTIFACTS
=====================================================================

1. Canonical JSON:
    <stem>/z_matrix.json

2. NPZ sidecar (unless --no-sidecar):
    <stem>/z_matrix.npz
    - Must be byte-reproducible under fixed SOURCE_DATE_EPOCH.
    - Must contain:
        Z or Z_sym
        neuron_order
        source_json_sha256
        numpy_version

3. Diagnostic plot (if --plot):
    <stem>/z_matrix.png
    AND a GUI popup window showing the heatmap.

4. Save-data file (if --save-data):
    <stem>/z_matrix.data.json

=====================================================================
README UPDATES
=====================================================================

Please update README.md to include:

- Description of Phase 02 inputs and outputs.
- Explanation of unified weight computation.
- Explanation of JSON + NPZ dual-artifact design.
- Full CLI usage block with all options:
    --plot
    --symmetric
    --normalize
    --config-hash
    --no-sidecar
    --strict
    --save-data
    --stats
- Example command lines.
- Example heatmap image description.
- Explanation of how Phase 03 loads NPZ.
- Explanation of how symmetric mode affects eigen-decomposition.
- Explanation of the GUI popup vs terminal summary.

=====================================================================
POST-RUN SUMMARY (TERMINAL ONLY)
=====================================================================

After running build_square_matrix.py, print a terminal summary block:

+--------------------------------------------------------------+
| Phase 02 complete                                            |
| Z matrix shape: NxN                                          |
| Unified weights: OK                                          |
| Symmetric mode: <yes/no>                                     |
| Normalization: <mode>                                        |
| JSON written: z_matrix.json                                  |
| NPZ written:  z_matrix.npz (unless --no-sidecar)             |
| Plot:        z_matrix.png (if --plot)                        |
| Save-data:   z_matrix.data.json (if --save-data)             |
| Stats: min/max/mean/std printed above                        |
+--------------------------------------------------------------+

=====================================================================
GUI POPUP (ONLY WHEN --plot IS USED)
=====================================================================

- Display the heatmap in a matplotlib window via plt.show().
- This is the ONLY GUI popup in Phase 02.
- The terminal summary remains separate.

=====================================================================

Please update the Phase 02 design document and generate the full
build_square_matrix.py implementation with these features.