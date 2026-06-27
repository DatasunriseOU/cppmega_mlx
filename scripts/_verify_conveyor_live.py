#!/usr/bin/env python3
"""SELECTIVE live-output verification (~200 samples) across CODE / COMMIT / BUILD-FILE.

For each sample: input_ids -> DETOKENIZE (CppMegaTokenizer) -> STRIP our special
tokens (BOS/PAD/FIM/<SPACE>/<NL> -> real ws, language-header comments removed for
clang-format) -> clang-format CODE (or json-format compile_commands). Verify NL(47)
present, roundtrip (reencode_idempotent), clang-format compilable-ish (distinguish
missing-include from corruption), sidecar per family A/B/C/D, COMMIT PR-as-docstring,
BUILD-file tagging, dedup. Writes rendered samples + per-type reports. READ-ONLY data.
"""
from __future__ import annotations
import glob, json, os, re, sqlite3, subprocess, sys, hashlib, random, collections
from pathlib import Path

ROOT = Path('/Volumes/external/sources/cppmega.mlx')
sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

CLANG_FORMAT = '/opt/homebrew/opt/llvm/bin/clang-format'
OUT = ROOT / 'outputs' / 'verification_report'
SAMP = OUT / 'conveyor_samples'
SAMP.mkdir(parents=True, exist_ok=True)

tok = load_cppmega_tokenizer(ROOT / 'cppmega_mlx' / 'tokenizer')
NL_ID = 47; SPACE_ID = 46; BOS_ID = 2; PAD_ID = 0
FIM_IDS = {4, 5, 6, 45}
SPECIAL_TEXT_RE = re.compile(r'<(?:BOS|EOS|PAD|UNK|FIM_[A-Z]+|CODE_(?:START|END)|FILE_SEP|DIFF_(?:START|END)|COMMENT_(?:START|END)|RESERVED_\d+)>')
LANG_RE = re.compile(r'language:\s*primary=(\S+)')
STD_RE = re.compile(r'standard=(\S+)')
PLAT_RE = re.compile(r'platform:\s*(\S+)')
PR_RE = re.compile(r'@pr\s+(\S+)')
SHA_RE = re.compile(r'@sha\s+(\S+)')
REPO_RE = re.compile(r'@repo\s+(\S+)')
DISC_RE = re.compile(r'@discussion', re.I)
DOCSTRING_RE = re.compile(r'/\*\*.*?\*/', re.DOTALL)
PRE_MARK = '=== PRE-COMMIT'; POST_MARK = '=== POST-COMMIT'
BUILD_LANGS = {'cmake', 'make', 'makefile', 'bazel', 'meson', 'ninja', 'compile_commands', 'build'}

SIDE_CHANNELS = {
 'A_platform': ['platform_ids', 'token_platform_ids'],
 'B_structure': ['token_structure_ids', 'token_dep_levels', 'token_ast_depth', 'token_sibling_index', 'token_ast_node_type'],
 'C_graph': ['token_symbol_ids', 'token_def_use', 'token_call_targets', 'token_type_refs', 'token_call_edges', 'token_type_edges'],
 'D_commit': ['token_change_mask_pre', 'token_change_mask_post', 'hunk_id_per_token', 'edit_op_per_token', 'changed_chunk_ids', 'changed_chunk_spans'],
}
ALL_SIDE = [c for v in SIDE_CHANNELS.values() for c in v]

def extract_git_history_process_health():
    """Report real JSONL writer duplicates, ignoring transient git subprocess forks.

    On macOS a subprocess fork can briefly inherit the parent's command line
    before exec'ing git. A raw ps grouping by --output sees that as a duplicate
    extract_git_history.py, but it is parented by the real extract process and
    is not a JSONL writer. Treat only non-extract-parent processes as writers.
    """
    try:
        out = subprocess.check_output(
            ['ps', '-axo', 'pid,ppid,stat,etime,command'],
            text=True,
        )
    except Exception as e:
        return {'error': str(e)}

    proc_re = re.compile(r'\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.*)')
    output_re = re.compile(r'--output\s+(\S+)')
    cmd_by_pid = {}
    rows = []
    for line in out.splitlines():
        m = proc_re.match(line)
        if not m:
            continue
        pid = int(m.group(1))
        ppid = int(m.group(2))
        cmd = m.group(5)
        cmd_by_pid[pid] = cmd
        if 'extract_git_history.py' not in cmd:
            continue
        out_m = output_re.search(cmd)
        rows.append({
            'pid': pid,
            'ppid': ppid,
            'stat': m.group(3),
            'etime': m.group(4),
            'output': out_m.group(1) if out_m else '',
            'line': line,
        })

    raw_by_output = collections.defaultdict(list)
    writer_by_output = collections.defaultdict(list)
    ignored_children = []
    for row in rows:
        raw_by_output[row['output']].append(row)
        parent_cmd = cmd_by_pid.get(row['ppid'], '')
        if 'extract_git_history.py' in parent_cmd:
            ignored_children.append(row)
        else:
            writer_by_output[row['output']].append(row)

    raw_dupes = {
        output: [row['line'] for row in group]
        for output, group in raw_by_output.items()
        if len(group) > 1
    }
    writer_dupes = {
        output: [row['line'] for row in group]
        for output, group in writer_by_output.items()
        if len(group) > 1
    }
    return {
        'raw_extract_outputs': len(raw_by_output),
        'raw_extract_procs': sum(len(group) for group in raw_by_output.values()),
        'raw_dupe_outputs': len(raw_dupes),
        'root_writer_outputs': len(writer_by_output),
        'root_writer_procs': sum(len(group) for group in writer_by_output.values()),
        'root_writer_dupe_outputs': len(writer_dupes),
        'fork_before_exec_children_ignored': len(ignored_children),
        'raw_dupe_examples': dict(list(raw_dupes.items())[:5]),
        'root_writer_dupe_examples': dict(list(writer_dupes.items())[:5]),
    }

def strip_for_clang(text: str) -> str:
    """Strip leading language/platform header comments + special-token text; keep code."""
    text = SPECIAL_TEXT_RE.sub('', text)
    lines = text.split('\n'); out = []; skipping = True
    for ln in lines:
        s = ln.strip()
        if skipping and (s.startswith('// language:') or s.startswith('// platform:') or
                         s.startswith('// compiler:') or s.startswith('// standard:') or
                         s.startswith('// arch:') or s.startswith('// mode:') or
                         s == '//' or s == '' or s.startswith('// <')):
            continue
        skipping = False; out.append(ln)
    return '\n'.join(out)

def clang_format(code: str):
    if not code.strip():
        return '', False, 'empty'
    fn = 'cc.json' if code.lstrip().startswith('[') or code.lstrip().startswith('{') else 'ex.cpp'
    style = '--style=LLVM' if fn.endswith('.cpp') else '--style=LLVM'
    p = subprocess.run([CLANG_FORMAT, f'--assume-filename={fn}', style],
                       input=code, capture_output=True, text=True)
    return p.stdout, p.returncode == 0, p.stderr.strip()[:200]

def try_compile(code: str):
    """Compile-ish: distinguish missing-include from corruption. Returns (ok, kind, msg)."""
    p = subprocess.run(['/opt/homebrew/opt/llvm/bin/clang', '-std=c++17', '-fsyntax-only',
                        '-x', 'c++', '-', '-I/opt/homebrew/include'],
                       input=code, capture_output=True, text=True, timeout=30)
    if p.returncode == 0:
        return True, 'compiles', ''
    err = p.stderr
    # missing include / undeclared identifier from missing headers = NOT corruption
    miss = ('file not found' in err or 'fatal error:' in err and 'expected' not in err.split('fatal error:')[1][:60]
            or 'use of undeclared' in err or 'unknown type name' in err or 'no member named' in err
            or 'no template named' in err or "'fmt/" in err or 'incomplete type' in err)
    corrupt = ('expected' in err and ('expected unqualified-id' in err or 'expected \';\'' in err
               or 'expected expression' in err or "expected '}'" in err or "expected '('" in err))
    if corrupt and not miss:
        return False, 'corrupt', err.strip()[:300]
    return False, 'missing-include', err.strip()[:200]

def fill_pct(row, ids_len):
    """Per-channel fill % = fraction of non-zero/non-empty entries (token-aligned) or presence (list)."""
    res = {}
    for c in ALL_SIDE:
        v = row.get(c)
        if v is None:
            res[c] = 0.0; continue
        vv = list(v)
        if c in ('token_call_edges', 'token_type_edges', 'changed_chunk_ids', 'changed_chunk_spans', 'platform_ids'):
            res[c] = 100.0 if len(vv) > 0 else 0.0
        else:
            if not vv:
                res[c] = 0.0; continue
            nz = sum(1 for x in vv if x not in (0, -1, None))
            res[c] = round(100.0 * nz / len(vv), 1)
    return res

def split_commit_blocks(text):
    blocks = {}
    m = DOCSTRING_RE.search(text)
    if m: blocks['docstring'] = m.group(0)
    pre = text.find(PRE_MARK); post = text.find(POST_MARK)
    blocks['has_pre'] = pre != -1; blocks['has_post'] = post != -1
    blocks['has_diff'] = ('diff --git' in text or '=== DIFF' in text)
    if m and pre != -1:
        blocks['doc_before_pre'] = m.start() < pre
    return blocks

def process(parquet, row_idx, forced_type=None):
    t = pq.read_table(parquet)
    row = t.slice(row_idx, 1).to_pylist()[0]
    ids = list(row['input_ids'])
    text = tok.decode(ids)
    head = tok.decode(ids[:120])
    lang_m = LANG_RE.search(head); std_m = STD_RE.search(head); plat_m = PLAT_RE.search(head)
    primary = lang_m.group(1) if lang_m else None
    # classify
    is_commit = (PRE_MARK in text) or (bool(row.get('token_change_mask_pre')) and any(row.get('token_change_mask_pre') or []))
    is_build = primary is not None and primary.lower() in BUILD_LANGS
    typ = 'COMMIT' if is_commit else ('BUILD-FILE' if is_build else 'CODE')
    # roundtrip
    re_ids = tok.encode(text); text2 = tok.decode(re_ids); re_ids2 = tok.encode(text2)
    reencode_idem = (re_ids == re_ids2)
    text_rt = (text == text2)
    # NL present
    nl_present = NL_ID in ids
    # clang-format
    code = strip_for_clang(text)
    fmt, cf_ok, cf_err = clang_format(code)
    comp_ok, comp_kind, comp_msg = (False, 'skip', '')
    if typ != 'COMMIT':  # compile-ish on code/build snippets (commit blocks are partial diffs)
        try:
            comp_ok, comp_kind, comp_msg = try_compile(code)
        except Exception as e:
            comp_ok, comp_kind, comp_msg = False, 'error', str(e)[:120]
    fills = fill_pct(row, len(ids))
    cblocks = split_commit_blocks(text) if is_commit else {}
    docstring = cblocks.get('docstring', '') or (DOCSTRING_RE.search(text).group(0) if DOCSTRING_RE.search(text) else '')
    rec = {
        'parquet': str(parquet), 'row': row_idx, 'type': typ,
        'n_tokens': len(ids), 'primary_lang': primary,
        'standard': std_m.group(1) if std_m else None,
        'platform': plat_m.group(1) if plat_m else None,
        'nl_present': nl_present, 'nl_count': ids.count(NL_ID),
        'reencode_idempotent': reencode_idem, 'text_roundtrip': text_rt,
        'clang_format_ok': cf_ok, 'clang_format_err': cf_err if not cf_ok else '',
        'compile_kind': comp_kind, 'compile_ok': comp_ok, 'compile_msg': comp_msg if comp_kind == 'corrupt' else '',
        'fills': fills,
        'has_docstring': bool(docstring),
        'pr_number': PR_RE.search(docstring).group(1) if PR_RE.search(docstring) else None,
        'sha_in_doc': SHA_RE.search(docstring).group(1) if SHA_RE.search(docstring) else None,
        'repo_in_doc': REPO_RE.search(docstring).group(1) if REPO_RE.search(docstring) else None,
        'has_discussion': bool(DISC_RE.search(docstring)) or bool(DISC_RE.search(text[:2000])),
        'commit_blocks': {k: v for k, v in cblocks.items() if k != 'docstring'},
        'repo': row.get('repo'), 'commit_hash': row.get('commit_hash'),
    }
    # dedup signal: hash of stripped code body (first 4000 chars of normalized code)
    body = re.sub(r'\s+', ' ', code).strip()[:4000]
    rec['_body_hash'] = hashlib.sha1(body.encode('utf-8', 'replace')).hexdigest()
    rec['_text'] = text; rec['_fmt'] = fmt; rec['_docstring'] = docstring; rec['_code'] = code
    return rec

# ---- build sample plan: TOKEN-MASS-STRATIFIED selection ----
# Rationale: the conveyor writes length-bucketed parquet (1024/2048/4096/8192/16384).
# The old "random.shuffle(files) then take the first-N CODE rows" oversampled the
# numerous tiny 1024-bucket leaf docs and crowded out the large 4096+ dependency-pack
# docs, so per-token side-channels (e.g. token_dep_levels) read artificially low
# (~2.1%) — an artifact of WHICH rows were drawn, not of the data. Fix: enumerate
# parquet per (type, length-bucket), compute each bucket's TOKEN-MASS (sum of
# valid_token_count), allocate the N-sample budget across buckets PROPORTIONAL to
# token-mass (so a 4096 dependency-pack bucket holding most of the corpus tokens
# gets most of the samples), then draw rows uniformly at random within each bucket.
# Per-type minimums keep CODE / COMMIT / BUILD all covered. READ-ONLY +
# concurrent-writer tolerant (skip FileNotFoundError / ArrowInvalid, re-raise else).
random.seed(1234)
code_files = sorted(glob.glob(str(ROOT / 'outputs/reindexed/*/*.parquet')))
commit_files = sorted(glob.glob(str(ROOT / 'outputs/reindexed_commits/*/*.parquet')))

# Configurable total sample budget (default ~300), split across the three types by
# token-mass share with per-type floors.
N_TOTAL = int(os.environ.get('VERIFY_N', '300'))
MIN_CODE = int(os.environ.get('VERIFY_MIN_CODE', '120'))
MIN_COMMIT = int(os.environ.get('VERIFY_MIN_COMMIT', '70'))
# BUILD-FILE rows are SPARSE in the C/C++ source corpus (build files are a tiny
# fraction of all docs and live in the small length buckets). The BUILD floor is the
# guaranteed minimum we MUST surface; it is found via a cheap head-prefilter scan
# (see sample_type(prefilter=True)). Keep it modest + achievable for the live corpus.
MIN_BUILD = int(os.environ.get('VERIFY_MIN_BUILD', '15'))

samples = []
skipped = []
counts = collections.Counter()
seen_keys = set()  # (parquet, row) guard against double-draw

def bucket_of(p):
    return Path(p).parent.name

def file_token_mass(p):
    """Token-mass (sum of valid_token_count) + row count for a parquet, concurrent-safe.
    Returns (n_rows, token_mass) or None if the file is mid-write (race)."""
    try:
        pf = pq.ParquetFile(p)
        nrows = pf.metadata.num_rows
        if nrows == 0:
            return (0, 0)
        col = pf.read(columns=['valid_token_count']).column('valid_token_count').to_pylist()
        return (nrows, int(sum(v for v in col if v is not None)))
    except (FileNotFoundError, ArrowInvalid) as e:
        skipped.append((p, 'token_mass', type(e).__name__)); return None

def enumerate_buckets(files):
    """Group files by length-bucket; aggregate (rows, token_mass) per bucket.
    Returns {bucket: {'files': [...], 'rows': int, 'mass': int}} (skips raced files)."""
    buckets = {}
    for p in files:
        tm = file_token_mass(p)
        if tm is None:
            continue
        nrows, mass = tm
        if nrows == 0:
            continue
        b = bucket_of(p)
        d = buckets.setdefault(b, {'files': [], 'rows': 0, 'mass': 0})
        d['files'].append((p, nrows, mass)); d['rows'] += nrows; d['mass'] += mass
    return buckets

def allocate_by_mass(buckets, budget):
    """Allocate `budget` samples across buckets proportional to token-mass.
    Largest-remainder rounding; never allocate more than a bucket has rows."""
    total_mass = sum(d['mass'] for d in buckets.values())
    if total_mass <= 0 or budget <= 0:
        return {b: 0 for b in buckets}
    raw = {b: budget * d['mass'] / total_mass for b, d in buckets.items()}
    alloc = {b: min(int(raw[b]), buckets[b]['rows']) for b in buckets}
    # distribute leftover by largest fractional remainder (respecting row caps)
    leftover = budget - sum(alloc.values())
    order = sorted(buckets, key=lambda b: (raw[b] - int(raw[b])), reverse=True)
    i = 0
    while leftover > 0 and order:
        b = order[i % len(order)]
        if alloc[b] < buckets[b]['rows']:
            alloc[b] += 1; leftover -= 1
        else:
            order.remove(b); i -= 1
            if not order: break
        i += 1
    return alloc

def classify_head(ids):
    """CHEAP type classification from the 120-token head only (no clang-format / compile
    probe). Mirrors process()'s classification. Returns 'COMMIT'/'BUILD-FILE'/'CODE'."""
    head = tok.decode(ids[:120])
    if PRE_MARK in head:
        return 'COMMIT'
    m = LANG_RE.search(head)
    prim = m.group(1).lower() if m else None
    if prim is not None and prim in BUILD_LANGS:
        return 'BUILD-FILE'
    return 'CODE'

def collect_sparse_floor(files, type_filter, floor):
    """DETERMINISTIC enumeration scan to satisfy a per-type FLOOR for a SPARSE type
    (e.g. BUILD-FILE: ~0.2% of rows). Random sampling with replacement is hopeless for
    needles in a haystack, so we read each parquet's input_ids column ONCE, cheaply
    head-classify every row (classify_head), and run the expensive process() ONLY on
    rows whose head matches `type_filter`, until `floor` are collected. Rows are visited
    in a fixed (shuffled-once) order for reproducibility. Concurrent-write tolerant.
    Returns number collected."""
    got = 0
    flist = list(files); random.shuffle(flist)
    for p in flist:
        if got >= floor:
            break
        try:
            col = pq.read_table(p, columns=['input_ids']).column('input_ids')
            nrows = len(col)
        except (FileNotFoundError, ArrowInvalid) as e:
            skipped.append((p, 'sparse_scan', type(e).__name__)); continue
        order = list(range(nrows)); random.shuffle(order)
        for ri in order:
            if got >= floor:
                break
            key = (p, ri)
            if key in seen_keys:
                continue
            try:
                ids = list(col[ri].as_py())
            except (FileNotFoundError, ArrowInvalid) as e:
                skipped.append((p, ri, 'sparse_head', type(e).__name__)); continue
            if classify_head(ids) != type_filter:
                continue  # cheap reject, no full process()
            try:
                rec = process(p, ri)
            except (FileNotFoundError, ArrowInvalid) as e:
                skipped.append((p, ri, type(e).__name__)); continue
            if rec['type'] != type_filter:
                continue
            seen_keys.add(key)
            samples.append(rec); counts[rec['type']] += 1; got += 1
    return got

def draw_bucket(bucket_files, want, type_filter, max_attempts_mult=8):
    """Draw up to `want` rows of `type_filter` from a bucket's files, weighted across
    files by token-mass, rows uniform-random within a file. Concurrent-safe.
    Returns number drawn. Used for ABUNDANT types (CODE/COMMIT) where uniform-random
    draws hit the wanted type almost every attempt. SPARSE types (BUILD-FILE) instead
    use collect_sparse_floor (enumeration), since random draws can't find needles."""
    got = 0
    if want <= 0 or not bucket_files:
        return 0
    fmass = [m for (_p, _n, m) in bucket_files]
    fnames = [p for (p, _n, _m) in bucket_files]
    if sum(fmass) <= 0:
        return 0
    attempts = 0
    max_attempts = want * max_attempts_mult + 50
    while got < want and attempts < max_attempts:
        attempts += 1
        p = random.choices(fnames, weights=fmass, k=1)[0]
        try:
            nrows = pq.ParquetFile(p).metadata.num_rows
        except (FileNotFoundError, ArrowInvalid) as e:
            skipped.append((p, 'draw', type(e).__name__)); continue
        if nrows == 0:
            continue
        ri = random.randrange(nrows)
        key = (p, ri)
        if key in seen_keys:
            continue
        try:
            rec = process(p, ri)
        except (FileNotFoundError, ArrowInvalid) as e:
            skipped.append((p, ri, type(e).__name__)); continue
        if type_filter and rec['type'] != type_filter:
            continue
        seen_keys.add(key)
        samples.append(rec); counts[rec['type']] += 1; got += 1
    return got

def sample_type(files, budget, type_filter, floor=0):
    """Token-mass-stratified draw across length buckets for an ABUNDANT type.

    Allocates `budget` samples across length buckets PROPORTIONAL to token-mass, then
    draws rows uniformly at random within each bucket (representative-by-token-mass).
    `floor`: if >0, this many samples of `type_filter` MUST be found or we RAISE
    (RULE #1 fail-loud: a per-type minimum that silently goes unmet is forbidden)."""
    buckets = enumerate_buckets(files)
    if not buckets:
        if floor > 0:
            raise RuntimeError(f"sample_type({type_filter}): NO parquet buckets found "
                               f"for {files[:2]}... but floor={floor} required")
        return 0
    alloc = allocate_by_mass(buckets, budget)
    total = 0
    for b, d in sorted(buckets.items(), key=lambda kv: kv[1]['mass'], reverse=True):
        total += draw_bucket(d['files'], alloc.get(b, 0), type_filter)
    # Top up from the highest-mass buckets if a bucket ran short on the wanted type.
    if total < budget:
        for b, d in sorted(buckets.items(), key=lambda kv: kv[1]['mass'], reverse=True):
            if total >= budget:
                break
            total += draw_bucket(d['files'], budget - total, type_filter)
    if floor > 0 and total < floor:
        raise RuntimeError(
            f"sample_type({type_filter}): per-type floor NOT met "
            f"(got {total} < floor {floor}) after mass-stratified draw across "
            f"{sum(d['rows'] for d in buckets.values())} rows — investigate "
            f"(do NOT silently lower the floor).")
    return total

# Token-mass share between CODE-stream (code+build live in outputs/reindexed) and the
# COMMIT-stream (outputs/reindexed_commits). Split the global budget by their share,
# then enforce per-type floors. CODE and BUILD-FILE both come from the CODE stream.
code_buckets = enumerate_buckets(code_files)
commit_buckets = enumerate_buckets(commit_files)
code_mass = sum(d['mass'] for d in code_buckets.values())
commit_mass = sum(d['mass'] for d in commit_buckets.values())
grand_mass = max(code_mass + commit_mass, 1)

# COMMIT budget: mass-share of the global N, floored at MIN_COMMIT.
commit_budget = max(MIN_COMMIT, round(N_TOTAL * commit_mass / grand_mass))
# Remaining budget goes to the CODE stream (CODE + BUILD-FILE), with floors.
code_stream_budget = max(MIN_CODE + MIN_BUILD, N_TOTAL - commit_budget)
build_budget = MIN_BUILD
code_budget = max(MIN_CODE, code_stream_budget - build_budget)

# COMMIT (own stream, abundant -> mass-stratified random draw, floor-checked).
sample_type(commit_files[:], commit_budget, 'COMMIT', floor=MIN_COMMIT)
# BUILD-FILE (sparse -> deterministic enumeration scan over the CODE stream to
# guarantee the floor; RAISE if genuinely unmet). Drawn BEFORE CODE so seen_keys
# excludes them from the CODE pass.
build_got = collect_sparse_floor(code_files[:], 'BUILD-FILE', MIN_BUILD)
if build_got < MIN_BUILD:
    raise RuntimeError(
        f"BUILD-FILE floor NOT met (got {build_got} < floor {MIN_BUILD}) after a full "
        f"deterministic enumeration scan of the CODE stream — the live corpus does not "
        f"yet contain {MIN_BUILD} build-classified rows (cmake/make/bazel/...). "
        f"Investigate the conveyor's build-file ingestion; do NOT silently lower the "
        f"floor (RULE #1: fail loud).")
# CODE (abundant -> token-mass-stratified random draw; this is the channel-fill
# representativeness path that kills the old first-N-shuffled artifact).
sample_type(code_files[:], code_budget, 'CODE', floor=MIN_CODE)

# ---- dedup: query db + spot-check duplicate bodies ----
dedup_info = {}
db = ROOT / 'outputs/dedup_seen.sqlite'
try:
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = con.cursor()
    tbls = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    dedup_info['tables'] = tbls
    for tb in tbls:
        try:
            c = cur.execute(f'SELECT COUNT(*) FROM "{tb}"').fetchone()[0]
            dedup_info[f'count_{tb}'] = c
        except Exception as e:
            dedup_info[f'count_{tb}'] = f'err:{e}'
    con.close()
except Exception as e:
    dedup_info['error'] = str(e)

body_hashes = collections.Counter(s['_body_hash'] for s in samples if s['type'] in ('CODE', 'BUILD-FILE'))
dups = {h: c for h, c in body_hashes.items() if c > 1}
dedup_info['sampled_code_bodies'] = sum(1 for s in samples if s['type'] in ('CODE', 'BUILD-FILE'))
dedup_info['exact_dup_body_groups'] = len(dups)
dedup_info['exact_dup_examples'] = list(dups.items())[:5]

# ---- write rendered samples ----
written = 0
for k, s in enumerate(samples):
    fn = SAMP / f"{s['type'].lower().replace('-','_')}_{k:03d}.md"
    md = []
    md.append(f"# {s['type']} sample {k} — {Path(s['parquet']).parent.name}/{Path(s['parquet']).name} row {s['row']}")
    md.append(f"- tokens: {s['n_tokens']}  primary_lang: {s['primary_lang']}  standard: {s['standard']}  platform: {s['platform']}")
    md.append(f"- NL(47) present: {s['nl_present']} (count={s['nl_count']})  reencode_idempotent: {s['reencode_idempotent']}  text_roundtrip: {s['text_roundtrip']}")
    md.append(f"- clang_format_ok: {s['clang_format_ok']}  compile_kind: {s['compile_kind']}  compile_ok: {s['compile_ok']}")
    if s['compile_msg']: md.append(f"  - CORRUPTION msg: {s['compile_msg']}")
    if s['type'] == 'COMMIT':
        md.append(f"- PR#: {s['pr_number']}  @sha: {s['sha_in_doc']}  @repo: {s['repo_in_doc']}  @discussion: {s['has_discussion']}  blocks: {s['commit_blocks']}")
        if s['_docstring']:
            md.append("## PR-as-docstring (HEAD, before PRE/POST/diff)\n```c\n" + s['_docstring'][:2500] + "\n```")
    md.append("## detok->strip->clang-format CODE\n```cpp\n" + s['_fmt'].strip()[:3500] + "\n```")
    md.append("## sidecar fill %\n```json\n" + json.dumps(s['fills'], indent=1) + "\n```")
    fn.write_text('\n'.join(md))
    written += 1

# ---- per-type aggregation ----
def agg(typ):
    ss = [s for s in samples if s['type'] == typ]
    if not ss: return {'n': 0}
    n = len(ss)
    chfill = {c: round(sum(s['fills'][c] for s in ss) / n, 1) for c in ALL_SIDE}
    return {
        'n': n,
        'nl_present_pct': round(100 * sum(s['nl_present'] for s in ss) / n, 1),
        'reencode_idempotent_pct': round(100 * sum(s['reencode_idempotent'] for s in ss) / n, 1),
        'text_roundtrip_pct': round(100 * sum(s['text_roundtrip'] for s in ss) / n, 1),
        'clang_format_ok_pct': round(100 * sum(s['clang_format_ok'] for s in ss) / n, 1),
        'compiles_pct': round(100 * sum(s['compile_ok'] for s in ss) / n, 1),
        'missing_include_pct': round(100 * sum(s['compile_kind'] == 'missing-include' for s in ss) / n, 1),
        'corrupt_pct': round(100 * sum(s['compile_kind'] == 'corrupt' for s in ss) / n, 1),
        'has_docstring_pct': round(100 * sum(s['has_docstring'] for s in ss) / n, 1),
        'pr_number_pct': round(100 * sum(bool(s['pr_number']) for s in ss) / n, 1),
        'discussion_pct': round(100 * sum(s['has_discussion'] for s in ss) / n, 1),
        'channel_fill_pct': chfill,
        'primary_langs': dict(collections.Counter(s['primary_lang'] for s in ss)),
    }

def bucket_report(buckets):
    return {b: {'files': len(d['files']), 'rows': d['rows'], 'token_mass': d['mass']}
            for b, d in sorted(buckets.items())}

sampling_plan = {
    'method': ('CODE/COMMIT: token-mass-stratified across length buckets (budget '
               'allocated per bucket by sum(valid_token_count), rows drawn uniformly '
               'within bucket). BUILD-FILE: deterministic enumeration scan (sparse '
               'type) to guarantee the per-type floor.'),
    'N_total': N_TOTAL,
    'per_type_floors': {'CODE': MIN_CODE, 'COMMIT': MIN_COMMIT, 'BUILD-FILE': MIN_BUILD},
    'budgets': {'CODE': code_budget, 'BUILD-FILE': build_budget, 'COMMIT': commit_budget},
    'code_stream_token_mass': code_mass,
    'commit_stream_token_mass': commit_mass,
    'code_buckets': bucket_report(code_buckets),
    'commit_buckets': bucket_report(commit_buckets),
}

report = {
    'total_samples': len(samples),
    'counts': dict(counts),
    'sampling_plan': sampling_plan,
    'process_health': extract_git_history_process_health(),
    'skipped_race': skipped,
    'per_type': {t: agg(t) for t in ('CODE', 'COMMIT', 'BUILD-FILE')},
    'dedup': dedup_info,
    'samples_written': written,
    'samples_dir': str(SAMP),
}
(OUT / 'live_verification_report.json').write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
