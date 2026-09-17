## Locked decisions

| # | Decision |
|---|---|
| **D-0 (new)** | `--plot` is **unchanged**: PNG always written; plain `plt.show()` popup on a GUI backend; still skipped on non-interactive backends; `--no-popup` still applies to it. `--interactive` is purely additive. When **both** are given, `main()` passes `popup=False` to `write_artifact_set(...)` (via `popup=not args.no_popup and not args.interactive`) and logs an INFO line, so exactly one window opens — the cursor-enabled one. `plot_matrix` itself is never touched. |
| D-1 | `--interactive` is an independent presentation-only flag that writes no artifact. |
| D-2 | Tooltip always shows the 4 required lines; the `raw unified weight` line is conditional on `--normalize`/`--symmetric`. |
| D-3 | Missing `mplcursors` → error (exit 1); non-interactive backend → warning (exit 0); `--dry-run` → no window. |
| D-4 | `--no-popup` affects `--plot` only; an explicitly requested `--interactive` window still opens (INFO logged). |
| D-5 | The `Interactive:` line in the summary box appears **only** when `--interactive` is used → the default box stays byte-identical. |
| D-6 | `plot_matrix` is not refactored into a shared figure builder (≈10 duplicated lines in `build_interactive_figure`) to keep `--plot` bit-for-bit unchanged. |

## What the patch will touch (final scope)

* **Modify:** `src/matrices/build_square_matrix.py` — module docstring/CLI text, `__all__`, `render_summary_box` (2 additive keyword-only params), `build_argument_parser` (`--interactive` in the `outputs` group), `main` (popup suppression + window invocation + 2 warnings), plus **6 new symbols** inserted between `plot_matrix` and `write_artifact_set`: `INTERACTIVE_ANNOTATION_KWARGS`, `_raw_weight_lookup`, `format_cell_tooltip`, `build_interactive_figure`, `attach_cell_cursor`, `show_interactive_matrix`.
* **Modify:** `src/matrices/__init__.py` — mirror the 5 new public names (TYPE_CHECKING + `__all__`).
* **Modify:** `environment.yml` — `- mplcursors>=0.7` under Visualisation; install `conda install -c conda-forge mplcursors`.
* **Modify:** `tests/test_build_square_matrix.py` — 15 new tests in a `# --interactive (hover tooltips)` section (all headless-safe under the pinned `Agg` backend; `pytest.importorskip("mplcursors")` only where a real cursor is needed).
* **Modify (docs):** `README.md` (usage block, options table, new `### Interactive mode (--interactive)` section with the `--plot` vs `--interactive` comparison + example commands, GUI-popup bullet, Environment, Status) and `execution-plans/02_build_square_matrix.md` (§4, §11, §14, new §16.1, §19, §21, §22, §24, §26).
* **Untouched:** `unified_weight(s)`, `normalize_matrix`, `symmetrize`, `compute_diagnostics`, `build_z_matrix`, `load_z_matrix`, sidecar I/O, `save_data_payload`, `render_statistics`, `plot_matrix`, `write_artifact_set`, `variant_stem`, `artifact_paths`, `resolve_inputs`, `can_popup`, all schemas/keys, `MATRIX_CONFIG_FIELDS`/config hash/filenames, and every Phase 01 / `src/utils/io.py` file.

## Verification I'll run after implementing

1. `conda install -c conda-forge mplcursors`, then a smoke test of `mplcursors.cursor(im, hover=True, highlight=False)` against the real 113×113 matrix (confirms the 3.11.1 / 0.7.1 combination and that `Selection.index` is `(row, col)`).
2. `python -m pytest -m "not slow"` then the full suite.
3. Byte-identity check: a `--plot` run before vs. after the patch on the reference artifact (`z_matrix.json` + `z_matrix.npz` must be identical).
4. Manual GUI check on `qtagg`: `--interactive` hover text vs. known weights; `--plot --interactive` must open exactly one window with the PNG on disk; `--dry-run --interactive` must open nothing.

I've stopped here as instructed: **no files have been written or modified.** When you're happy with the plan, toggle to **Act mode** and I'll implement it in the order laid out in §13 (dependency → module → `__init__` → tests → docs → verification).