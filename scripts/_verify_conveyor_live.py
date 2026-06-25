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

# ---- build sample plan ----
random.seed(1234)
code_files = sorted(glob.glob(str(ROOT / 'outputs/reindexed/*/*.parquet')))
commit_files = sorted(glob.glob(str(ROOT / 'outputs/reindexed_commits/*/*.parquet')))

samples = []
skipped = []
counts = collections.Counter()
TARGET_COMMIT = 70; TARGET_BUILD = 30; TARGET_CODE = 110

def iter_rows(files, want, type_filter):
    got = 0
    random.shuffle(files)
    for p in files:
        if got >= want: break
        try:
            nrows = pq.ParquetFile(p).metadata.num_rows
        except (FileNotFoundError, ArrowInvalid) as e:
            skipped.append((p, type(e).__name__)); continue
        order = list(range(nrows)); random.shuffle(order)
        for ri in order:
            if got >= want: break
            try:
                rec = process(p, ri)
            except (FileNotFoundError, ArrowInvalid) as e:
                skipped.append((p, ri, type(e).__name__)); continue
            if type_filter and rec['type'] != type_filter:
                continue
            samples.append(rec); counts[rec['type']] += 1; got += 1
            yield rec

list(iter_rows(commit_files[:], TARGET_COMMIT, 'COMMIT'))
list(iter_rows(code_files[:], TARGET_BUILD, 'BUILD-FILE'))
list(iter_rows(code_files[:], TARGET_CODE, 'CODE'))

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

report = {
    'total_samples': len(samples),
    'counts': dict(counts),
    'skipped_race': skipped,
    'per_type': {t: agg(t) for t in ('CODE', 'COMMIT', 'BUILD-FILE')},
    'dedup': dedup_info,
    'samples_written': written,
    'samples_dir': str(SAMP),
}
(OUT / 'live_verification_report.json').write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
