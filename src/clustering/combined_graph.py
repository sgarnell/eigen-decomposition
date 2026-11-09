#!/usr/bin/env python3
"""Combined graph exporter — fully CLI-driven.

Joins a Phase 01 parsed graph (neuron nodes + directed pairs with pre_z/post_z)
to ONE Phase 04 run (groups, motifs, families, pathway edges) and writes a
single GraphML file in yEd's dialect:
  - neuron nodes nested inside group (tile) nodes,
  - group titles carry the associated families (distinct-neuron counts; the
    home family -- the family of the group's dominant mode -- in parentheses),
  - pair edges keep pre_z / post_z / w as data, colored red (pre-dominance)
    to blue (post-dominance),
  - pathway edges group-to-group, thick, dashed, off-gradient (gold default).
  - resonator candidates (with --resonators): reciprocal neuron pairs scored
    S = G x E -- G the geometric mean of the two |w| (loop strength), E the
    geometric mean of the two input shares (the partner's |w| as a fraction
    of each neuron's total incoming |w|, i.e. how much of what a neuron
    hears is the other neuron) -- drawn as dotted green neuron-to-neuron
    edges, arrows at both ends, labeled with S, carrying S/G/E as data.
    Transmitter grading (brighter colour for ACh<->Glu pairs) resolves
    in this order: an explicit option (at most one of
    --transmitters-neuprint / --transmitters-json '{"NEURON": "x", ...}'
    / the legacy --transmitters CSV); else a transmitters.json file
    beside this script (or in the working directory); else neuPrint,
    when NEUPRINT_TOKEN is set — a failed fetch only skips the grading;
    else no grading. The export succeeds in every case.
Self-loops are kept by default; --no-self-loops suppresses them.

Run it from the repo root; every path resolves from the repo layout and can be
overridden individually:

    python combined_graph.py --dataset <dataset_name> --config-hash 87ef7c02 --preview
    python combined_graph.py --dataset <dataset_name> --runs-dir /tmp/sweep/rt050 --config-hash 87ef7c02
    python combined_graph.py --dataset <dataset_name> --motifs path/to/motifs.x.y.data.json --preview

--dataset is REQUIRED: there is no built-in dataset name, so the tool works
for any user and any future dataset.
"""
import argparse, glob, json, math, os, sys
from xml.sax.saxutils import escape

import numpy as np

TILE_BORDER = "#8A97A5"
NEURON_FILL = "#DFE7EF"


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_inputs(a):
    """Return (parsed_path, z_matrix_path, motifs_path) honouring overrides."""
    proc_dir = a.processed_dir or os.path.join(a.root, "data", "processed", a.dataset)
    parsed = a.parsed or os.path.join(proc_dir, "parsed_graph.json")
    if a.z_matrix:
        zmat = a.z_matrix
    else:
        cands = sorted(glob.glob(os.path.join(proc_dir, "z_matrix.*.npz")))
        if len(cands) != 1:
            fail(f"expected exactly one z_matrix.*.npz in {proc_dir}, found {len(cands)}; use --z-matrix")
        zmat = cands[0]
    if a.motifs:
        motifs = a.motifs
    else:
        runs_dir = a.runs_dir or os.path.join(a.root, "outdir")
        cands = sorted(glob.glob(os.path.join(runs_dir, a.dataset, "motifs.*.data.json")))
        if a.config_hash:
            cands = [c for c in cands if a.config_hash in os.path.basename(c)]
        if len(cands) != 1:
            listing = "\n  ".join(cands) if cands else "(none)"
            fail(f"need exactly one motifs data.json in {os.path.join(runs_dir, a.dataset)}"
                 f"{' for hash ' + a.config_hash if a.config_hash else ''}, found {len(cands)}:\n  {listing}\n"
                 "use --config-hash or --motifs")
        motifs = cands[0]
    for label, path in (("parsed", parsed), ("z-matrix", zmat), ("motifs", motifs)):
        if not os.path.isfile(path):
            fail(f"{label} file not found: {path}")
    return parsed, zmat, motifs


def lerp(c1, c2, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(c1, c2))


def w_color(w, wmin, wmax):
    red, mid, blue = (229, 72, 77), (150, 140, 170), (77, 163, 255)
    t = 0.0 if wmax == wmin else (w - wmin) / (wmax - wmin)
    rgb = lerp(red, mid, t * 2) if t < 0.5 else lerp(mid, blue, (t - 0.5) * 2)
    return "#%02X%02X%02X" % rgb


NT_ALIASES = {"acetylcholine": "ach", "ach": "ach", "glutamate": "glu", "glu": "glu",
              "dopamine": "da", "da": "da", "gaba": "gaba",
              "serotonin": "5-ht", "5-ht": "5-ht", "octopamine": "oa"}


def load_transmitters(path):
    """CSV with header neuron,transmitter -> {neuron: normalized NT code}."""
    import csv
    table = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("neuron") or "").strip()
            raw = (row.get("transmitter") or "").strip().lower()
            if name and raw:
                table[name] = NT_ALIASES.get(raw, raw)
    return table


def _transmitters_from_obj(obj):
    """Parsed JSON (object or list of objects) -> {neuron: NT code}.
    Raises ValueError on a shape that is neither."""
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = ((r.get("neuron"), r.get("transmitter")) for r in obj
                 if isinstance(r, dict))
    else:
        raise ValueError('must be a JSON object {"NEURON": "transmitter"} '
                         'or a list of {"neuron": ..., "transmitter": ...} '
                         "objects")
    table = {}
    for name, raw in items:
        name = (name or "").strip()
        raw = (raw or "").strip().lower()
        if name and raw:
            table[name] = NT_ALIASES.get(raw, raw)
    return table


def load_transmitters_json(text):
    """JSON string -> {neuron: normalized NT code}. The table is small
    and stable, so it can live inline in the command line / a script
    rather than in a separate CSV file."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        fail(f"--transmitters-json is not valid JSON: {e}")
    try:
        return _transmitters_from_obj(obj)
    except ValueError as e:
        fail(f"--transmitters-json {e}")


def load_transmitters_json_file(path):
    """JSON file (same shapes as --transmitters-json) -> NT table."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        fail(f"cannot read transmitters file {path}: {e}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        fail(f"{path} is not valid JSON: {e}")
    try:
        return _transmitters_from_obj(obj)
    except ValueError as e:
        fail(f"{path} {e}")


NEUPRINT_URL = "https://neuprint.janelia.org"


def neuprint_query(cypher, token, dataset, limit=100000):
    """POST one custom Cypher query to neuPrint (stdlib only), mirroring
    the transport in ephaptic-coupling/scripts/neuprint_client.py.
    Returns a list of row dicts."""
    import urllib.request, urllib.error
    payload = {"cypher": cypher, "dataset": dataset, "version": "0.5.0",
               "args": [], "limit": int(limit)}
    req = urllib.request.Request(
        NEUPRINT_URL + "/api/custom/custom",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json",
                 "Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"neuPrint query failed: HTTP {e.code} — check "
                           "NEUPRINT_TOKEN and --neuprint-dataset")
    except urllib.error.URLError as e:
        raise RuntimeError(f"neuPrint query failed: {e.reason}")
    rows = body.get("data") if isinstance(body, dict) else body
    cols = body.get("columns") if isinstance(body, dict) else None
    if isinstance(rows, list) and rows and isinstance(rows[0], list) and cols:
        return [dict(zip(cols, r)) for r in rows]
    return rows or []


def neuron_type_candidates(name):
    """Instance name -> neuPrint type candidates, most specific first:
    'FB4B(NO2)' -> ['FB4B']; 'hDeltaH3' -> ['hDeltaH3', 'hDeltaH']."""
    base = name.split("(")[0].strip()
    cands = [base]
    stripped = base.rstrip("0123456789")
    if stripped and stripped != base:
        cands.append(stripped)
    return cands


def load_transmitters_neuprint(names, token, dataset):
    """Fetch predicted transmitters for the run's neurons from neuPrint.

    One Cypher query over every candidate type. Each type's call is
    neuPrint's own cell-type prediction (celltypePredictedNt) when the
    type carries one, else the majority predictedNt across its neurons.
    Returns {instance name: normalized NT code} and prints the resolved
    table as a JSON string, so the run log doubles as the copy-paste
    form for --transmitters-json."""
    from collections import Counter
    wanted = set()
    for n in names:
        wanted.update(neuron_type_candidates(n))
    type_list = ", ".join("'" + t.replace("'", "\\'") + "'" for t in sorted(wanted))
    cypher = ("MATCH (n:Neuron) WHERE n.type IN [" + type_list + "] "
              "RETURN n.type AS type, n.celltypePredictedNt AS cellNt, "
              "n.predictedNt AS nt")
    rows = neuprint_query(cypher, token, dataset)
    by_type = {}
    for r in rows:
        slot = by_type.setdefault(r.get("type"),
                                  {"cell": Counter(), "nt": Counter()})
        cell = (r.get("cellNt") or "").strip().lower()
        nt = (r.get("nt") or "").strip().lower()
        if cell:
            slot["cell"][cell] += 1
        if nt:
            slot["nt"][nt] += 1
    type_call = {}
    for t, slot in by_type.items():
        votes = slot["cell"] or slot["nt"]
        if votes:
            type_call[t] = votes.most_common(1)[0][0]
    table, raw, unresolved = {}, {}, []
    for n in names:
        hit = next((type_call[c] for c in neuron_type_candidates(n)
                    if c in type_call), None)
        if hit:
            raw[n] = hit
            table[n] = NT_ALIASES.get(hit, hit)
        else:
            unresolved.append(n)
    print(f"neuprint: resolved {len(raw)}/{len(names)} neurons from {dataset}; "
          f"unresolved: {', '.join(unresolved) if unresolved else 'none'}")
    print("transmitters-json: " + json.dumps(raw, sort_keys=True))
    return table


def resolve_transmitters(a, order):
    """The transmitter table used for grading, in priority order:

    1. an explicit option (--transmitters-neuprint / --transmitters-json
       / --transmitters) — at most one may be given;
    2. a transmitters.json file beside this script, else in the CWD;
    3. neuPrint, but only when NEUPRINT_TOKEN is set — and any failure
       there is a warning, never a failed run;
    4. no table: grading is skipped and every candidate stays the
       standard colour. The run succeeds in all cases."""
    sources = [bool(a.transmitters_neuprint), bool(a.transmitters_json),
               bool(a.transmitters)]
    if sum(sources) > 1:
        fail("use only one transmitter source: --transmitters-neuprint, "
             "--transmitters-json or --transmitters")
    if a.transmitters_neuprint:
        token = os.environ.get("NEUPRINT_TOKEN", "").strip()
        if not token:
            fail("NEUPRINT_TOKEN environment variable is required for "
                 "neuPrint access")
        try:
            return load_transmitters_neuprint(order, token, a.neuprint_dataset)
        except RuntimeError as e:
            fail(str(e))
    if a.transmitters_json:
        return load_transmitters_json(a.transmitters_json)
    if a.transmitters:
        return load_transmitters(a.transmitters)
    for cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "transmitters.json"),
                 os.path.join(os.getcwd(), "transmitters.json")):
        if os.path.isfile(cand):
            print(f"transmitters: {cand}")
            return load_transmitters_json_file(cand)
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if token:
        try:
            print("transmitters: neuPrint (auto — NEUPRINT_TOKEN is set)")
            return load_transmitters_neuprint(order, token, a.neuprint_dataset)
        except RuntimeError as e:
            print(f"warning: neuPrint fetch failed ({e}); "
                  "ACh-Glu grading skipped")
            return {}
    print("transmitters: none found (no transmitters.json beside the "
          "script, no NEUPRINT_TOKEN); ACh-Glu grading skipped")
    return {}


def find_resonators(pairs, pair_ws, min_score, nt):
    """Reciprocal pairs scored S = G x E. Returns (candidates, n_reciprocal).

    G = sqrt(|w_ab * w_ba|)            -- loop strength (weaker leg dominates)
    E = sqrt(share_b_from_a * share_a_from_b), share = |w| / total incoming |w|
    """
    wmap = {(p["source"], p["target"]): w for p, w in zip(pairs, pair_ws)}
    in_abs = {}
    for (s, t), w in wmap.items():
        in_abs[t] = in_abs.get(t, 0.0) + abs(w)
    cands, seen, n_recip = [], set(), 0
    for (s, t), w in wmap.items():
        if s == t or (t, s) in seen or (t, s) not in wmap:
            continue
        seen.add((s, t))
        n_recip += 1
        w2 = wmap[(t, s)]
        G = math.sqrt(abs(w * w2))
        E = (math.sqrt((abs(w) / in_abs[t]) * (abs(w2) / in_abs[s]))
             if in_abs.get(s) and in_abs.get(t) else 0.0)
        S = G * E
        if S < min_score:
            continue
        pattern = ""
        if {nt.get(s), nt.get(t)} == {"ach", "glu"}:
            pattern = "ACh-Glu"
        cands.append({"a": s, "b": t, "w_ab": w, "w_ba": w2,
                      "G": G, "E": E, "S": S, "nt": pattern})
    cands.sort(key=lambda r: -r["S"])
    return cands, n_recip


def build_titles(data):
    """gid -> title. Distinct-neuron counts; home family in parentheses."""
    fam_of_mode, fam_label = {}, {}
    for fam in data["families"]:
        label = fam.get("label") or f"F{fam['family_id']:02d}"
        fam_label[fam["family_id"]] = label
        for m in fam["modes"]:
            fam_of_mode[int(m)] = label
    motif_members = {int(m["mode"]): {mem["neuron_id"] for mem in m["members"]}
                     for m in data["motifs"]}
    titles = {}
    for g in data["groups"]:
        gid = g["group_id"]
        if g.get("is_background"):
            titles[gid] = f"{gid} · background"
            continue
        members = set(g["members"])
        counts = {}
        for fam in data["families"]:
            label = fam_label[fam["family_id"]]
            fam_neurons = set()
            for m in fam["modes"]:
                fam_neurons |= motif_members.get(int(m), set())
            counts[label] = len(members & fam_neurons)
        home_mode = g.get("dominant_mode")
        home = fam_of_mode.get(int(home_mode)) if home_mode else None
        others = sorted(((l, c) for l, c in counts.items() if l != home and c > 0),
                        key=lambda kv: (-kv[1], kv[0]))
        title = f"{gid} · Mode {home_mode} · ({home})"
        if others:
            title += " · " + ", ".join(f"{l}×{c}" for l, c in others)
        titles[gid] = title
    return titles


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the combined neuron+group graph (yEd GraphML) from a "
                    "Phase 01 parsed graph and one Phase 04 run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inp = ap.add_argument_group("inputs")
    inp.add_argument("--root", default=".", help="repo root (data/ and outdir/ live under it)")
    inp.add_argument("--dataset", required=True,
                     help="dataset name (its folder under data/processed/ and the runs dir); "
                          "required — no dataset name is built in")
    inp.add_argument("--processed-dir", default=None, help="dir holding parsed_graph.json (default: <root>/data/processed/<dataset>)")
    inp.add_argument("--parsed", default=None, help="explicit parsed_graph.json (overrides --processed-dir)")
    inp.add_argument("--z-matrix", default=None, help="explicit z_matrix.*.npz (default: the one in the processed dir)")
    inp.add_argument("--runs-dir", default=None, help="dir containing <dataset>/ run folders (default: <root>/outdir)")
    inp.add_argument("--motifs", default=None, help="explicit motifs.*.data.json (overrides --runs-dir/--config-hash)")
    inp.add_argument("--config-hash", default=None, help="select the run whose filename contains this config hash")
    out = ap.add_argument_group("outputs")
    out.add_argument("--out", default=None, help="GraphML path (default: beside the run, combined.<hash>.graphml)")
    out.add_argument("--preview", action="store_true", help="also render a PNG preview")
    out.add_argument("--png-out", default=None, help="PNG path (default: --out with .png)")
    out.add_argument("--title", default=None, help="preview title (default: built from the run's config)")
    sty = ap.add_argument_group("style / content")
    sty.add_argument("--self-loops", action=argparse.BooleanOptionalAction, default=True,
                     help="keep self-loop pair edges (--no-self-loops suppresses them)")
    sty.add_argument("--group-edge-color", default="#E8A33D")
    sty.add_argument("--group-edge-width", type=float, default=3.5)
    sty.add_argument("--tile-fill", default="#232936")
    sty.add_argument("--resonators", action="store_true",
                     help="add the resonator-candidate layer (dotted green edges between "
                          "reciprocal neuron pairs, scored S = G x E)")
    sty.add_argument("--resonator-min-score", type=float, default=0.05,
                     help="minimum S for a pair to enter the resonator layer")
    sty.add_argument("--resonator-color", default="#2EE66B")
    sty.add_argument("--resonator-color-strong", default="#C6FF4D",
                     help="colour for candidates matching the ACh<->Glu transmitter pattern "
                          "(requires --transmitters)")
    sty.add_argument("--transmitters-neuprint", action="store_true",
                     help="fetch transmitter predictions for the run's neurons live "
                          "from neuPrint (dataset --neuprint-dataset); the API token is "
                          "read from the NEUPRINT_TOKEN environment variable, as in "
                          "ephaptic-coupling's mito_downloader")
    sty.add_argument("--neuprint-dataset", default="hemibrain:v1.2.1",
                     help="neuPrint dataset for --transmitters-neuprint")
    sty.add_argument("--transmitters-json", default=None,
                     help="optional JSON string mapping neuron -> transmitter, e.g. "
                          '\'{"FB5R": "glutamate", "hDeltaH3": "acetylcholine"}\'; used only '
                          "to grade resonator candidates; pairs without data stay the "
                          "standard colour")
    sty.add_argument("--transmitters", default=None,
                     help="legacy: optional CSV file (neuron,transmitter) with the same "
                          "role as --transmitters-json; use one or the other, not both")
    a = ap.parse_args(argv)

    parsed_path, zmat_path, motifs_path = resolve_inputs(a)
    parsed = json.load(open(parsed_path))
    data = json.load(open(motifs_path))
    cfg = data.get("config", {})
    cfg_hash = cfg.get("config_hash") or os.path.basename(motifs_path).split(".")[2]
    grouping = cfg.get("grouping", "?")
    rel_t = cfg.get("relative_threshold", "?")
    print(f"parsed : {parsed_path}")
    print(f"motifs : {motifs_path}  (grouping={grouping}, relative-threshold={rel_t}, config={cfg_hash})")
    print(f"z-matrix: {zmat_path}")

    out_path = a.out or os.path.join(os.path.dirname(motifs_path), f"combined.{cfg_hash}.graphml")
    z = np.load(zmat_path)
    W = z["matrix"]
    order = [str(x) for x in z["neuron_order"]]
    widx = {n: i for i, n in enumerate(order)}
    pairs = parsed["pairs"]
    pair_ws = [float(W[widx[p["source"]], widx[p["target"]]]) for p in pairs]
    wmin, wmax = min(pair_ws), max(pair_ws)

    titles = build_titles(data)
    groups = data["groups"]
    neuron_group = {n: g["group_id"] for g in groups for n in g["members"]}

    resonators, n_recip = [], 0
    if a.resonators:
        nt = resolve_transmitters(a, order)
        resonators, n_recip = find_resonators(pairs, pair_ws, a.resonator_min_score, nt)

    # ---- layout (yEd pixels; a hierarchical pass in yEd reorganizes anyway) ----
    NW, NH, GAPX, GAPY, PAD, TITLEH = 118, 30, 14, 12, 16, 44
    boxes, npos = {}, {}
    gx, gy, rowh = 40, 40, 0
    for g in groups:
        n = max(1, len(g["members"]))
        cols = max(1, math.ceil(math.sqrt(n * 1.9)))
        rows = math.ceil(len(g["members"]) / cols) if g["members"] else 1
        bw = PAD * 2 + cols * NW + (cols - 1) * GAPX
        bh = TITLEH + PAD + rows * NH + (rows - 1) * GAPY + PAD // 2
        if gx + bw > 1900:
            gx, gy, rowh = 40, gy + rowh + 70, 0
        boxes[g["group_id"]] = (gx, gy, bw, bh)
        rowh = max(rowh, bh)
        for i, name in enumerate(g["members"]):
            r, c = divmod(i, cols)
            npos[name] = (gx + PAD + c * (NW + GAPX) + NW / 2,
                          gy + TITLEH + PAD + r * (NH + GAPY) + NH / 2)
        gx += bw + 60

    def neuron_node_xml(name):
        gx0, gy0, _, _ = boxes[neuron_group[name]]
        cx, cy = npos[name]
        x, y = cx - gx0 - NW / 2, cy - gy0 - NH / 2
        return (f'<node id="n{widx[name]}">'
                f'<data key="k_group">{escape(neuron_group[name])}</data>'
                f'<data key="d5"><y:ShapeNode><y:Geometry height="{NH}" width="{NW}" x="{x:.1f}" y="{y:.1f}"/>'
                f'<y:Fill color="{NEURON_FILL}" transparent="false"/><y:BorderStyle color="#5A6672" type="line" width="1.0"/>'
                f'<y:NodeLabel alignment="center" fontSize="11" textColor="#10131A">{escape(name)}</y:NodeLabel>'
                f'<y:Shape type="roundrectangle"/></y:ShapeNode></data></node>')

    L = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
             'xmlns:java="http://www.yworks.com/xml/yfiles-common/1.0/java" '
             'xmlns:sys="http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0" '
             'xmlns:x="http://www.yworks.com/xml/yfiles-common/markup/2.0" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xmlns:y="http://www.yworks.com/xml/graphml" '
             'xmlns:yed="http://www.yworks.com/xml/yed/3" '
             'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
             'http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">')
    L.append('<key id="d5" for="node" yfiles.type="nodegraphics"/>')
    L.append('<key id="d9" for="edge" yfiles.type="edgegraphics"/>')
    L.append('<key id="k_group" for="node" attr.name="group" attr.type="string"/>')
    L.append('<key id="k_title" for="node" attr.name="title" attr.type="string"/>')
    L.append('<key id="k_kind" for="edge" attr.name="kind" attr.type="string"/>')
    L.append('<key id="k_pre" for="edge" attr.name="pre_z" attr.type="double"/>')
    L.append('<key id="k_post" for="edge" attr.name="post_z" attr.type="double"/>')
    L.append('<key id="k_w" for="edge" attr.name="w" attr.type="double"/>')
    L.append('<key id="k_weight" for="edge" attr.name="weight" attr.type="double"/>')
    L.append('<key id="k_rs" for="edge" attr.name="resonator_S" attr.type="double"/>')
    L.append('<key id="k_rg" for="edge" attr.name="resonator_G" attr.type="double"/>')
    L.append('<key id="k_re" for="edge" attr.name="resonator_E" attr.type="double"/>')
    L.append('<key id="k_rnt" for="edge" attr.name="nt_pattern" attr.type="string"/>')
    L.append('<graph edgedefault="directed" id="G">')
    for g in groups:
        gid = g["group_id"]
        x, y, bw, bh = boxes[gid]
        L.append(f'<node id="{gid}"><data key="k_title">{escape(titles[gid])}</data>'
                 f'<data key="d5"><y:GroupNode><y:Geometry height="{bh}" width="{bw}" x="{x}" y="{y}"/>'
                 f'<y:Fill color="{a.tile_fill}" transparent="false"/>'
                 f'<y:BorderStyle color="{TILE_BORDER}" type="line" width="1.5"/>'
                 f'<y:NodeLabel alignment="center" fontSize="13" fontStyle="bold" textColor="#E8ECF1">{escape(titles[gid])}</y:NodeLabel>'
                 f'<y:Shape type="rectangle"/><y:State closed="false"/></y:GroupNode></data>'
                 f'<graph edgedefault="directed" id="{gid}:">')
        for name in g["members"]:
            L.append(neuron_node_xml(name))
        L.append('</graph></node>')
    nloops = 0
    for i, p in enumerate(pairs):
        if p["source"] == p["target"]:
            nloops += 1
            if not a.self_loops:
                continue
        w = pair_ws[i]
        L.append(f'<edge id="p{i}" source="n{widx[p["source"]]}" target="n{widx[p["target"]]}">'
                 f'<data key="k_kind">pair</data>'
                 f'<data key="k_pre">{p["pre_z"]}</data><data key="k_post">{p["post_z"]}</data>'
                 f'<data key="k_w">{w:.4f}</data>'
                 f'<data key="d9"><y:PolyLineEdge><y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>'
                 f'<y:LineStyle color="{w_color(w, wmin, wmax)}" type="line" width="1.0"/>'
                 f'<y:Arrows source="none" target="standard"/><y:BendStyle smoothed="false"/>'
                 f'</y:PolyLineEdge></data></edge>')
    for j, e in enumerate(data["pathway"]["edges"]):
        wt = float(e["weight"])
        lab = f'{e["source"]}→{e["target"]} {wt:+.2f}'
        L.append(f'<edge id="g{j}" source="{e["source"]}" target="{e["target"]}">'
                 f'<data key="k_kind">group</data><data key="k_weight">{wt:.4f}</data>'
                 f'<data key="d9"><y:PolyLineEdge><y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>'
                 f'<y:LineStyle color="{a.group_edge_color}" type="dashed" width="{a.group_edge_width}"/>'
                 f'<y:Arrows source="none" target="standard"/>'
                 f'<y:EdgeLabel fontSize="12" fontStyle="bold" textColor="{a.group_edge_color}">{escape(lab)}</y:EdgeLabel>'
                 f'<y:BendStyle smoothed="false"/></y:PolyLineEdge></data></edge>')
    for k, r in enumerate(resonators):
        strong = r["nt"] == "ACh-Glu"
        col = a.resonator_color_strong if strong else a.resonator_color
        wid = 2.8 if strong else 2.0
        lab = f'S {r["S"]:.3f}' + (" · ACh-Glu" if strong else "")
        L.append(f'<edge id="r{k}" source="n{widx[r["a"]]}" target="n{widx[r["b"]]}">'
                 f'<data key="k_kind">resonator</data>'
                 f'<data key="k_rs">{r["S"]:.4f}</data><data key="k_rg">{r["G"]:.4f}</data>'
                 f'<data key="k_re">{r["E"]:.4f}</data><data key="k_rnt">{escape(r["nt"])}</data>'
                 f'<data key="d9"><y:PolyLineEdge><y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>'
                 f'<y:LineStyle color="{col}" type="dotted" width="{wid}"/>'
                 f'<y:Arrows source="standard" target="standard"/>'
                 f'<y:EdgeLabel fontSize="11" fontStyle="bold" textColor="{col}">{escape(lab)}</y:EdgeLabel>'
                 f'<y:BendStyle smoothed="false"/></y:PolyLineEdge></data></edge>')
    L.append('</graph></graphml>')
    with open(out_path, "w") as fh:
        fh.write("\n".join(L))
    print(f"graphml: {len(groups)} group nodes, {len(pairs)} pairs "
          f"({nloops} self-loops, {'kept' if a.self_loops else 'suppressed'}), "
          f"{len(data['pathway']['edges'])} group edges -> {out_path}")
    for g in groups:
        print(f"  {titles[g['group_id']]}  ({len(g['members'])} neurons)")
    if a.resonators:
        print(f"resonators: {len(resonators)} of {n_recip} reciprocal pairs with "
              f"S >= {a.resonator_min_score}")
        for i, r in enumerate(resonators, 1):
            print(f"  R{i} {r['a']} <-> {r['b']}: S={r['S']:.4f} G={r['G']:.3f} E={r['E']:.3f} "
                  f"w={r['w_ab']:+.2f}/{r['w_ba']:+.2f} "
                  f"groups {neuron_group[r['a']]}/{neuron_group[r['b']]}"
                  f"{'  [ACh-Glu]' if r['nt'] else ''}")

    if a.preview or a.png_out:
        png_path = a.png_out or os.path.splitext(out_path)[0] + ".png"
        title = a.title or (f"Combined graph — {grouping} grouping · "
                            f"relative-threshold {rel_t} · config {cfg_hash}")
        render_preview(png_path, data, groups, titles, boxes, npos, pairs, pair_ws,
                       wmin, wmax, title, a.group_edge_color,
                       resonators, a.resonator_color, a.resonator_color_strong)
    return 0


def render_preview(path, data, groups, titles, boxes, npos, pairs, pair_ws,
                   wmin, wmax, title, gold, resonators=(), res_color="#2EE66B",
                   res_color_strong="#C6FF4D"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

    BG, INK, DIM = "#000000", "#e8ecf1", "#9aa7b4"
    YO = 100
    boxes = {k: (x, y + YO, w, h) for k, (x, y, w, h) in boxes.items()}
    npos = {k: (x, y + YO) for k, (x, y) in npos.items()}
    xs = [b[0] + b[2] for b in boxes.values()]
    ys = [b[1] + b[3] for b in boxes.values()]
    Wpx, Hpx = max(xs) + 40, max(ys) + 130
    fig, ax = plt.subplots(figsize=(Wpx / 95, Hpx / 95), dpi=110)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_position([0, 0, 1, 1]); ax.set_xlim(0, Wpx); ax.set_ylim(Hpx, 0); ax.axis("off")
    ax.text(Wpx / 2, 28, title, size=19, color="#4da3ff", weight="bold", ha="center")
    subtitle = ("neuron pairs: thin, red pre-dominance → blue post-dominance · "
                "group pathway edges: thick dashed gold · rings: self-loops")
    if resonators:
        subtitle += " · resonator candidates: dotted green (S = G x E)"
    ax.text(Wpx / 2, 62, subtitle, size=12.5, color=DIM, ha="center")
    for p, w in zip(pairs, pair_ws):
        if p["source"] == p["target"]:
            continue
        (x1, y1), (x2, y2) = npos[p["source"]], npos[p["target"]]
        ax.plot([x1, x2], [y1, y2], color=w_color(w, wmin, wmax), lw=0.55, alpha=0.34, zorder=1)
    for p, w in zip(pairs, pair_ws):
        if p["source"] == p["target"]:
            x, y = npos[p["source"]]
            ax.add_patch(Circle((x + 66, y - 12), 9, fill=False,
                                color=w_color(w, wmin, wmax), lw=1.4, zorder=6))
    centers = {}
    for g in groups:
        gid = g["group_id"]
        x, y, bw, bh = boxes[gid]
        centers[gid] = (x + bw / 2, y + bh / 2)
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=10",
                                    fc="#171b24", ec="#8a97a5", lw=1.3, zorder=3))
        tparts = titles[gid].split(" · ")
        if len(tparts) > 3:
            ax.text(x + bw / 2, y + 20, " · ".join(tparts[:3]), size=11.5, color=INK,
                    weight="bold", ha="center", zorder=6)
            ax.text(x + bw / 2, y + 38, " · ".join(tparts[3:]), size=10.5, color=INK,
                    weight="bold", ha="center", zorder=6)
        else:
            ax.text(x + bw / 2, y + 26, titles[gid], size=11.5, color=INK, weight="bold",
                    ha="center", zorder=6)
        for name in g["members"]:
            cx, cy = npos[name]
            ax.add_patch(FancyBboxPatch((cx - 59, cy - 15), 118, 30,
                                        boxstyle="round,pad=0,rounding_size=7",
                                        fc="#dfe7ef", ec="#5a6672", lw=0.8, zorder=4))
            nsize = 8.6 if len(name) <= 15 else (7.4 if len(name) <= 20 else 6.7)
            ax.text(cx, cy, name, size=nsize, color="#10131a", ha="center", va="center", zorder=6)
    for e in data["pathway"]["edges"]:
        (x1, y1), (x2, y2) = centers[e["source"]], centers[e["target"]]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=19,
                                     color=gold, lw=3.2, linestyle=(0, (6, 3)), zorder=7,
                                     shrinkA=10, shrinkB=10, connectionstyle="arc3,rad=0.13"))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my - 8, f'{float(e["weight"]):+.2f}', size=11, color=gold, weight="bold",
                ha="center", zorder=8)
    for r in resonators:
        (x1, y1), (x2, y2) = npos[r["a"]], npos[r["b"]]
        col = res_color_strong if r["nt"] else res_color
        ax.plot([x1, x2], [y1, y2], color=col, lw=2.4 if r["nt"] else 1.9,
                linestyle=(0, (1.6, 2.4)), zorder=8)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 16, f'S {r["S"]:.3f}', size=9,
                color=col, weight="bold", ha="center", zorder=9)
    ax.text(Wpx / 2, Hpx - 24, "Preview of the GraphML content — final layout is yEd's hierarchical "
                               "pass; group edges are the filtered pathway set (0.3 mode + 0.7 raw block sums)",
            size=11.5, color=DIM, ha="center", style="italic")
    plt.savefig(path, facecolor=BG)
    print("preview:", path)


if __name__ == "__main__":
    sys.exit(main())
