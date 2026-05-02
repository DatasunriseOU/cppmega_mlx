"""Vendored from nanochat/nanochat/cpp_tokenizer.py for byte-exact CUDA-side decode parity.

Source: https://github.com/apstenku123/nanochat (commit 330dc714)
Do not edit. Re-vendor by copying the upstream file when nanochat updates.

Encode works via HuggingFace tokenizers library.
Decode uses custom space reconstruction (BERT-style: insert spaces between word tokens).
"""

import os
from tokenizers import Tokenizer


# Single-char punctuation set
_PUNCT = set("{}()[]<>;:,.+-*/%&|^~!?=#@$_\\\"'")
# Multi-char operators
_MULTI_OPS = {
    "::",
    "->",
    ".*",
    "->*",
    "++",
    "--",
    "##",
    "==",
    "!=",
    "<=",
    ">=",
    "<=>",
    "&&",
    "||",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
    "...",
    "//",
    "/*",
    "*/",
    "@@",
    "---",
    "+++",
    "a/",
    "b/",
}


class CppTokenizer:
    """Hybrid C++ tokenizer: fixed vocab + learned BPE, BERT-style whitespace."""

    def __init__(self, tokenizer_path: str):
        if os.path.isdir(tokenizer_path):
            tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # Build reverse map for decoding
        self._vocab = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        # Build set of added/fixed-vocab tokens for BPE subword detection.
        # Added tokens are standalone words (keywords, STL names, etc.) that
        # should always get spaces around them. BPE-learned tokens may be
        # subword fragments that should attach to the previous token.
        added = self._tokenizer.get_added_tokens_decoder()
        self._added_token_ids = frozenset(added.keys())

    # --- nanochat-compatible API ---

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_vocab_size(self):
        return self.vocab_size

    def get_special_tokens(self):
        # Return all angle-bracket tokens
        return [t for t in self._vocab if t.startswith("<") and t.endswith(">")]

    def encode_special(self, text):
        # Support both nanochat-style <|bos|> and our <BOS> format
        result = self._vocab.get(text, None)
        if result is None:
            # Map nanochat special tokens to our format
            mapping = {
                "<|bos|>": "<BOS>",
                "<|eos|>": "<EOS>",
                "<|pad|>": "<PAD>",
                "<|endoftext|>": "<BOS>",
                # Tool-calling tokens (nanochat-style aliases)
                "<|code_start|>": "<CODE_START>",
                "<|code_end|>": "<CODE_END>",
                "<|thought_start|>": "<THINK_START>",
                "<|thought_end|>": "<THINK_END>",
                "<|query_tool|>": "<QUERY_TOOL>",
                "<|tool_result|>": "<TOOL_RESULT>",
                # Phase 1 aliases
                "<|compile_start|>": "<COMPILE_START>",
                "<|compile_end|>": "<COMPILE_END>",
                # Phase 3 aliases
                "<|debug_context|>": "<DEBUG_CONTEXT>",
                # Phase 4 thinking token aliases
                "<|think_error|>": "<THINK_ERROR>",
                "<|think_fix|>": "<THINK_FIX>",
                "<|think_trace|>": "<THINK_TRACE>",
                "<|think_verify|>": "<THINK_VERIFY>",
                "<|think_plan|>": "<THINK_PLAN>",
                "<|think_end|>": "<THINK_END>",
                # Phase 5 aliases
                "<|script_start|>": "<SCRIPT_START>",
                "<|script_end|>": "<SCRIPT_END>",
            }
            mapped = mapping.get(text)
            if mapped:
                result = self._vocab.get(mapped, None)
        return result

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """Encode text or list of texts. Compatible with nanochat API."""
        prepend_id: int | None = None
        append_id: int | None = None
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self._tokenizer.encode(text).ids
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
            return ids
        elif isinstance(text, list):
            results = [enc.ids for enc in self._tokenizer.encode_batch(text)]
            if prepend is not None:
                for row in results:
                    row.insert(0, prepend_id)
            if append is not None:
                for row in results:
                    row.append(append_id)
            return results
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    def _is_bpe_suffix(
        self,
        token: str,
        token_id: int,
        prev_token: str | None = None,
        prev_id: int | None = None,
    ) -> bool:
        """Check if a token is a BPE suffix fragment (not a standalone word).

        Returns True for tokens like 'ype', 'nal', 'tion' that are never
        standalone C++ identifiers and always attach to the previous token.
        Conservative: only matches clear suffixes, not ambiguous short words
        like 'is', 'or', 'if' which could be standalone.

        When prev_token/prev_id are provided, uses context to make better
        decisions (e.g., 's' after 'char' is a suffix forming 'chars').
        """
        # Fixed-vocab (added) tokens are always standalone
        if token_id in self._added_token_ids:
            return False
        # Must be purely alphabetic
        if not token.isalpha():
            return False
        # Only lowercase tokens can be suffixes
        if not token.islower():
            return False
        # 1-2 char tokens: context-dependent suffix detection.
        # Common single-char variable names (n, x, i, c, b, f, etc.) must
        # NOT be treated as suffixes after type keywords (int n, char c).
        _COMMON_SINGLE_VARS = frozenset("abcdefghijklmnopqrstuvwxyz")
        _COMMON_TWO_CHAR_IDS = frozenset(
            {
                # Common 2-char variable names and words used standalone in code
                "is",
                "it",
                "if",
                "in",
                "or",
                "on",
                "to",
                "do",
                "no",
                "at",
                "by",
                "up",
                "an",
                "be",
                "he",
                "me",
                "my",
                "of",
                "so",
                "we",
                "us",
                "id",
                "ok",
                "fn",
                "go",
            }
        )
        if len(token) <= 2:
            if prev_id is None or prev_token is None or not prev_token[-1:].isalpha():
                return False
            if prev_id in self._added_token_ids:
                # After an added keyword: allow common suffix chars (s, d)
                # that form plurals/past tense (chars, ints, used, signed)
                if len(token) == 1 and token in "sd":
                    return True
                # 2-char: allow only when NOT a common standalone identifier
                if len(token) == 2 and token not in _COMMON_TWO_CHAR_IDS:
                    return True
                return False
            # After a BPE fragment (not an added keyword): likely mid-word
            if prev_token.isalpha():
                # But not if token looks like a common standalone variable/word
                if len(token) == 1 and token in _COMMON_SINGLE_VARS:
                    # Single char after a non-keyword word is ambiguous.
                    # Heuristic: treat as suffix if prev is a known BPE
                    # fragment (short, uncommon as standalone word)
                    return (
                        len(prev_token) <= 4 and prev_token not in _COMMON_TWO_CHAR_IDS
                    )
                if len(token) == 2 and token in _COMMON_TWO_CHAR_IDS:
                    return False
                return True
            return False
        # 3+ char tokens: only treat as suffix if they look like one
        # (not a common standalone word)
        _COMMON_SHORT_WORDS = frozenset(
            {
                # C++ keywords and common identifiers that happen to be short
                "add",
                "all",
                "and",
                "any",
                "arg",
                "bad",
                "bar",
                "big",
                "bit",
                "buf",
                "bus",
                "can",
                "cap",
                "car",
                "cmd",
                "col",
                "con",
                "cpu",
                "cur",
                "del",
                "dim",
                "dir",
                "dns",
                "doc",
                "dst",
                "dup",
                "end",
                "env",
                "err",
                "ext",
                "fan",
                "far",
                "fix",
                "fmt",
                "foo",
                "fun",
                "gap",
                "get",
                "got",
                "gpu",
                "has",
                "hex",
                "hit",
                "hot",
                "hub",
                "idx",
                "img",
                "inc",
                "ini",
                "key",
                "len",
                "lib",
                "log",
                "low",
                "map",
                "max",
                "mem",
                "mid",
                "min",
                "mix",
                "mod",
                "msg",
                "neg",
                "net",
                "new",
                "nil",
                "nop",
                "not",
                "now",
                "num",
                "obj",
                "odd",
                "old",
                "one",
                "opt",
                "ord",
                "out",
                "own",
                "pad",
                "par",
                "per",
                "pid",
                "pin",
                "pkg",
                "pop",
                "pos",
                "pre",
                "ptr",
                "put",
                "raw",
                "red",
                "ref",
                "reg",
                "rem",
                "rep",
                "req",
                "res",
                "ret",
                "rev",
                "rgb",
                "row",
                "run",
                "say",
                "sec",
                "set",
                "sim",
                "sin",
                "src",
                "std",
                "str",
                "sub",
                "sum",
                "syn",
                "sys",
                "tab",
                "tag",
                "tan",
                "tcp",
                "tmp",
                "top",
                "try",
                "tty",
                "two",
                "udp",
                "uid",
                "url",
                "use",
                "usr",
                "val",
                "var",
                "vec",
                "via",
                "vol",
                "was",
                "way",
                "web",
                "win",
                "xml",
                "xor",
                "yes",
                "zip",
                # Common English words (prevent false suffix detection in prose)
                "the",
                "for",
                "are",
                "but",
                "yet",
                "nor",
                "had",
                "his",
                "her",
                "its",
                "our",
                "who",
                "how",
                "may",
                "did",
                "let",
                "got",
                "ago",
                "few",
                "own",
                "too",
                "day",
                "see",
                "saw",
                "ran",
                "ask",
                "why",
                "also",
                "been",
                "does",
                "each",
                "goes",
                "just",
                "like",
                "made",
                "many",
                "much",
                "only",
                "same",
                "some",
                "such",
                "than",
                "them",
                "they",
                "very",
                "what",
                "when",
                "will",
                "your",
                # 4-char common words
                "auto",
                "back",
                "base",
                "bind",
                "body",
                "bool",
                "byte",
                "call",
                "case",
                "cast",
                "char",
                "code",
                "copy",
                "core",
                "ctrl",
                "data",
                "date",
                "dead",
                "deep",
                "diff",
                "disc",
                "done",
                "down",
                "drop",
                "dump",
                "edge",
                "edit",
                "else",
                "emit",
                "enum",
                "eval",
                "even",
                "exec",
                "exit",
                "expr",
                "face",
                "fail",
                "fast",
                "file",
                "fill",
                "find",
                "flag",
                "flat",
                "flip",
                "flow",
                "fold",
                "font",
                "fork",
                "form",
                "free",
                "from",
                "full",
                "func",
                "glob",
                "good",
                "goto",
                "grid",
                "grow",
                "half",
                "halt",
                "hand",
                "hash",
                "have",
                "head",
                "heap",
                "help",
                "here",
                "hide",
                "high",
                "hint",
                "hold",
                "home",
                "hook",
                "host",
                "http",
                "huge",
                "info",
                "init",
                "into",
                "item",
                "iter",
                "join",
                "jump",
                "keep",
                "kern",
                "kill",
                "kind",
                "last",
                "lazy",
                "leaf",
                "left",
                "less",
                "line",
                "link",
                "list",
                "live",
                "load",
                "lock",
                "long",
                "look",
                "loop",
                "lost",
                "main",
                "make",
                "mark",
                "mask",
                "math",
                "menu",
                "meta",
                "mode",
                "more",
                "most",
                "move",
                "must",
                "mute",
                "name",
                "near",
                "need",
                "next",
                "node",
                "none",
                "norm",
                "note",
                "null",
                "open",
                "over",
                "pack",
                "page",
                "pair",
                "part",
                "pass",
                "past",
                "path",
                "peek",
                "pick",
                "ping",
                "pipe",
                "plan",
                "play",
                "plot",
                "plus",
                "poll",
                "pool",
                "port",
                "post",
                "prev",
                "proc",
                "prog",
                "prop",
                "pull",
                "pure",
                "push",
                "quit",
                "rand",
                "rank",
                "rate",
                "read",
                "real",
                "rect",
                "redo",
                "rest",
                "ring",
                "root",
                "rule",
                "safe",
                "save",
                "scan",
                "seed",
                "seek",
                "self",
                "send",
                "show",
                "shut",
                "side",
                "sign",
                "size",
                "skip",
                "slot",
                "slow",
                "snap",
                "sock",
                "sort",
                "spec",
                "spin",
                "sqrt",
                "stat",
                "stay",
                "step",
                "stop",
                "swap",
                "sync",
                "tail",
                "take",
                "task",
                "temp",
                "term",
                "test",
                "text",
                "that",
                "then",
                "this",
                "tick",
                "time",
                "tiny",
                "todo",
                "tone",
                "tool",
                "tree",
                "trim",
                "true",
                "turn",
                "type",
                "uint",
                "undo",
                "unit",
                "unix",
                "used",
                "user",
                "view",
                "void",
                "wait",
                "walk",
                "want",
                "warn",
                "wide",
                "with",
                "word",
                "work",
                "wrap",
                "zero",
                "zone",
                # 5-char common words (prevent false suffix in natural language)
                "about",
                "after",
                "again",
                "array",
                "async",
                "await",
                "begin",
                "being",
                "below",
                "block",
                "break",
                "build",
                "bytes",
                "cache",
                "catch",
                "chain",
                "check",
                "child",
                "chunk",
                "class",
                "clean",
                "clear",
                "close",
                "color",
                "const",
                "count",
                "cover",
                "crash",
                "debug",
                "defer",
                "delta",
                "depth",
                "dirty",
                "empty",
                "error",
                "event",
                "every",
                "exact",
                "extra",
                "false",
                "fetch",
                "field",
                "final",
                "first",
                "fixed",
                "flags",
                "flush",
                "focus",
                "force",
                "found",
                "frame",
                "front",
                "given",
                "graph",
                "green",
                "group",
                "guard",
                "guess",
                "happy",
                "hash5",
                "heavy",
                "hence",
                "image",
                "index",
                "inner",
                "input",
                "inter",
                "items",
                "known",
                "label",
                "large",
                "later",
                "layer",
                "level",
                "light",
                "limit",
                "local",
                "lower",
                "magic",
                "major",
                "match",
                "merge",
                "minor",
                "model",
                "mouse",
                "multi",
                "mutex",
                "never",
                "newer",
                "nodes",
                "occur",
                "often",
                "older",
                "order",
                "other",
                "outer",
                "owned",
                "owner",
                "param",
                "parse",
                "patch",
                "pause",
                "phase",
                "pixel",
                "place",
                "plain",
                "point",
                "power",
                "press",
                "print",
                "prior",
                "probe",
                "proof",
                "proxy",
                "query",
                "queue",
                "quick",
                "quiet",
                "quota",
                "raise",
                "range",
                "rapid",
                "ratio",
                "ready",
                "realm",
                "refer",
                "reply",
                "reset",
                "retry",
                "right",
                "round",
                "route",
                "scale",
                "scene",
                "scope",
                "serve",
                "setup",
                "shape",
                "share",
                "sharp",
                "shift",
                "short",
                "since",
                "sleep",
                "slice",
                "small",
                "smart",
                "space",
                "spawn",
                "split",
                "stack",
                "stage",
                "start",
                "state",
                "still",
                "store",
                "strip",
                "super",
                "table",
                "taken",
                "their",
                "there",
                "thing",
                "think",
                "those",
                "throw",
                "timer",
                "times",
                "title",
                "token",
                "total",
                "trace",
                "track",
                "trait",
                "tries",
                "tuple",
                "under",
                "union",
                "until",
                "upper",
                "using",
                "utils",
                "valid",
                "value",
                "watch",
                "where",
                "which",
                "while",
                "white",
                "whole",
                "width",
                "world",
                "would",
                "write",
                "yield",
                # 6-char common words
                "accept",
                "access",
                "action",
                "active",
                "actual",
                "affect",
                "always",
                "amount",
                "append",
                "assert",
                "assign",
                "atomic",
                "attach",
                "before",
                "better",
                "binary",
                "branch",
                "bridge",
                "broken",
                "bucket",
                "buffer",
                "bundle",
                "called",
                "cancel",
                "change",
                "client",
                "closed",
                "column",
                "commit",
                "common",
                "config",
                "create",
                "cursor",
                "custom",
                "daemon",
                "decode",
                "define",
                "delete",
                "deploy",
                "design",
                "detail",
                "detect",
                "device",
                "digest",
                "direct",
                "double",
                "driver",
                "during",
                "enable",
                "encode",
                "enough",
                "ensure",
                "entity",
                "equals",
                "escape",
                "except",
                "export",
                "extend",
                "extern",
                "failed",
                "family",
                "figure",
                "filter",
                "finish",
                "follow",
                "format",
                "friend",
                "frozen",
                "future",
                "gather",
                "global",
                "google",
                "gotten",
                "handle",
                "header",
                "height",
                "helper",
                "hidden",
                "ignore",
                "import",
                "inline",
                "insert",
                "inside",
                "invoke",
                "island",
                "itself",
                "launch",
                "layout",
                "length",
                "likely",
                "linear",
                "listen",
                "little",
                "loader",
                "locked",
                "logger",
                "lookup",
                "manage",
                "manual",
                "mapper",
                "margin",
                "marker",
                "master",
                "matrix",
                "member",
                "memory",
                "method",
                "middle",
                "module",
                "moment",
                "mostly",
                "native",
                "nested",
                "normal",
                "notice",
                "notify",
                "number",
                "object",
                "obtain",
                "offset",
                "online",
                "opener",
                "option",
                "origin",
                "output",
                "packet",
                "parent",
                "parser",
                "passed",
                "prefer",
                "public",
                "random",
                "reader",
                "reason",
                "record",
                "reduce",
                "reload",
                "remove",
                "render",
                "repair",
                "repeat",
                "report",
                "result",
                "resume",
                "retain",
                "return",
                "revert",
                "review",
                "rewind",
                "runner",
                "sample",
                "schema",
                "scroll",
                "search",
                "secure",
                "select",
                "sender",
                "server",
                "signal",
                "signed",
                "simple",
                "single",
                "sizeof",
                "socket",
                "source",
                "splice",
                "status",
                "stderr",
                "stdout",
                "stored",
                "stream",
                "string",
                "struct",
                "submit",
                "suffix",
                "switch",
                "symbol",
                "syntax",
                "system",
                "target",
                "thread",
                "throws",
                "toggle",
                "traits",
                "update",
                "upload",
                "vector",
                "verify",
                "weight",
                "widget",
                "window",
                "worker",
                "writer",
            }
        )
        if token in _COMMON_SHORT_WORDS:
            return False
        # Remaining 3-6 char lowercase BPE tokens are likely suffixes
        # (e.g., 'ype', 'nal', 'ern', 'tion', 'ment', 'ible', 'clude')
        if len(token) <= 6:
            return True
        return False

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text with BERT-style space reconstruction.

        Heuristic rules for C++ spacing. Output is approximate —
        use clang-format for exact formatting.
        """
        tokens = [self._id_to_token.get(i, "<UNK>") for i in ids]
        if not tokens:
            return ""

        # C++ type keywords — pointer/reference operators attach to these
        # without spaces: char*, int&, bool*, float&, etc.
        _TYPE_KEYWORDS = frozenset(
            {
                "void",
                "bool",
                "char",
                "short",
                "int",
                "long",
                "float",
                "double",
                "signed",
                "unsigned",
                "auto",
                "wchar_t",
                "char8_t",
                "char16_t",
                "char32_t",
                "size_t",
                "string",
                "vector",
                "map",
                "set",
                "list",
                "deque",
                "array",
                "pair",
                "tuple",
                "shared_ptr",
                "unique_ptr",
                "weak_ptr",
                "optional",
                "variant",
                "any",
            }
        )

        # Track whether we are inside a multi-token identifier (joined via
        # underscores). After an underscore join, all subsequent alphabetic
        # BPE fragments should continue joining (e.g., end + _ + po + int
        # = end_point). This is distinct from pure BPE suffix joins which
        # should not propagate to unrelated standalone words.
        in_underscore_id = False  # are we inside an underscore identifier?
        # Track whether the previous join was a short BPE suffix fragment
        # (1-2 chars like 'po', 'st'). After such a suffix, the next
        # alphabetic token is likely a continuation of the same word
        # (e.g., check+po+int = checkpoint).
        prev_short_suffix = False

        def need_space(prev, curr, prev_id, curr_id):
            nonlocal in_underscore_id, prev_short_suffix
            # Save and reset suffix tracking
            was_short_suffix = prev_short_suffix
            prev_short_suffix = False

            # Whitespace tokens: never add extra space
            if curr in ("\n", "\n\n") or prev in ("\n", "\n\n"):
                in_underscore_id = False
                return False
            # Special tokens: no space
            if curr.startswith("<") and curr.endswith(">") and len(curr) > 1:
                in_underscore_id = False
                return False
            if prev.startswith("<") and prev.endswith(">") and len(prev) > 1:
                in_underscore_id = False
                return False
            # Underscore attaches to adjacent word/identifier tokens.
            # The pre-tokenizer splits on _, so it appears as a separate
            # token between identifier parts: is + _ + prime -> is_prime
            # But _ should NOT attach to operators: "x = _" stays spaced.
            if curr == "_" and (prev[-1:].isalnum() or prev == "_"):
                in_underscore_id = True
                return False
            if prev == "_" and (curr[0:1].isalnum() or curr == "_"):
                in_underscore_id = True
                return False
            # Inside an underscore identifier (e.g., after end_), BPE
            # fragments that are part of the identifier must join.
            # end + _ + po + int = end_point (po and int are fragments)
            if in_underscore_id and curr.isalpha() and prev[-1:].isalpha():
                # Still inside the identifier: keep joining
                return False
            # No longer inside an underscore identifier
            in_underscore_id = False
            # BPE suffix continuation: attach to previous token when it
            # looks like a word fragment (e.g., size_t + ype -> size_type,
            # char + s -> chars)
            is_suffix = (
                self._is_bpe_suffix(curr, curr_id, prev, prev_id)
                and prev[-1:].isalpha()
            )
            if is_suffix:
                # Track short suffixes for the next iteration
                if len(curr) <= 2:
                    prev_short_suffix = True
                return False
            # After a short BPE suffix (like 'po'), the next alphabetic
            # token is likely a continuation of the same word, even if it's
            # an added keyword (e.g., check+po+int = checkpoint).
            if was_short_suffix and curr.isalpha() and prev[-1:].isalpha():
                return False
            # Attach operators (no space either side): :: -> .* ->* ++ -- ##
            if curr in ("::", "->", ".*", "->*", "++", "--", "##"):
                return False
            if prev in ("::", "->", ".*", "->*", "++", "--", "##"):
                return False
            # . always attaches (member access)
            if curr == "." or prev == ".":
                return False
            # No space after ( [ and no space before ) ] ; , .
            if prev in ("(", "["):
                return False
            if curr in (")", "]", ";", ","):
                return False
            # No space between word/keyword and (  e.g. main(, if(, for(
            if curr == "(":
                return False
            # No space between word/keyword and [  e.g. argv[], data[]
            if curr == "[":
                return False
            # Space before { (block opener)
            if curr == "{":
                return True
            # Template angle brackets:
            # No space before < after known template types: vector<int>
            # No space after < in templates: <int>
            # No space before > in templates: int>
            # But comparison operators (i < n) should keep spaces.
            _TEMPLATE_TYPES = frozenset(
                {
                    "vector",
                    "map",
                    "set",
                    "list",
                    "deque",
                    "array",
                    "pair",
                    "tuple",
                    "queue",
                    "stack",
                    "multimap",
                    "multiset",
                    "unordered_map",
                    "unordered_set",
                    "shared_ptr",
                    "unique_ptr",
                    "weak_ptr",
                    "optional",
                    "variant",
                    "function",
                    "basic_string",
                    "span",
                    "mdspan",
                    "expected",
                    "template",
                }
            )
            if curr == "<" and prev in _TEMPLATE_TYPES:
                return False
            if curr == ">" and prev_id in self._added_token_ids:
                # > after a type keyword is likely closing template: int>
                return False
            if prev == "<" and curr_id in self._added_token_ids:
                # < before a type keyword is likely opening template: <int
                return False
            # Pointer/reference operators attach to type keywords without space:
            # char* ptr, int& ref, const string& s
            # But keep space for binary usage: a * b, a & b
            if curr in ("*", "&") and prev in _TYPE_KEYWORDS:
                return False
            if curr in ("*", "&") and prev == "const":
                return False
            # Binary operators: always space around
            if curr in _MULTI_OPS or prev in _MULTI_OPS:
                return True
            # Single-char operators: = + - * / % & | ^ ~ ! ? < > #
            if curr in "=+-*/%&|^~!?<>#" or prev in "=+-*/%&|^~!?<>#":
                return True
            # Remaining: no space before : (label, access specifier)
            if curr == ":":
                return False
            if prev == ":":
                return True
            # Two word tokens: space
            return True

        parts = [tokens[0]]
        for i in range(1, len(tokens)):
            if need_space(tokens[i - 1], tokens[i], ids[i - 1], ids[i]):
                parts.append(" ")
            parts.append(tokens[i])

        return "".join(parts)

    def id_to_token(self, id: int) -> str:
        return self._id_to_token.get(id, "<UNK>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        return self._vocab.get("<BOS>", 2)

    @property
    def eos_token_id(self) -> int:
        return self._vocab.get("<EOS>", 3)

    @property
    def pad_token_id(self) -> int:
        return self._vocab.get("<PAD>", 0)

    # Tool-calling special token IDs
    @property
    def code_start_id(self) -> int:
        return self._vocab.get("<CODE_START>", 7)

    @property
    def code_end_id(self) -> int:
        return self._vocab.get("<CODE_END>", 8)

    @property
    def thought_start_id(self) -> int:
        return self._vocab.get("<THINK_START>", 9)

    @property
    def thought_end_id(self) -> int:
        return self._vocab.get("<THINK_END>", 10)

    @property
    def query_tool_id(self) -> int:
        return self._vocab.get("<QUERY_TOOL>", 11)

    @property
    def tool_result_id(self) -> int:
        return self._vocab.get("<TOOL_RESULT>", 19)

    # Phase 1 tokens
    @property
    def compile_start_id(self) -> int:
        return self._vocab.get("<COMPILE_START>", 29)

    @property
    def compile_end_id(self) -> int:
        return self._vocab.get("<COMPILE_END>", 30)

    # Phase 3 tokens
    @property
    def debug_context_id(self) -> int:
        return self._vocab.get("<DEBUG_CONTEXT>", 13)

    # Phase 4 thinking tokens
    @property
    def think_error_id(self) -> int:
        return self._vocab.get("<THINK_ERROR>", 21)

    @property
    def think_fix_id(self) -> int:
        return self._vocab.get("<THINK_FIX>", 22)

    @property
    def think_trace_id(self) -> int:
        return self._vocab.get("<THINK_TRACE>", 25)

    @property
    def think_verify_id(self) -> int:
        return self._vocab.get("<THINK_VERIFY>", 23)

    @property
    def think_plan_id(self) -> int:
        return self._vocab.get("<THINK_PLAN>", 24)

    @property
    def think_end_id(self) -> int:
        return self._vocab.get("<THINK_END>", 10)

    # Phase 5 tokens
    @property
    def script_start_id(self) -> int:
        return self._vocab.get("<SCRIPT_START>", 26)

    @property
    def script_end_id(self) -> int:
        return self._vocab.get("<SCRIPT_END>", 27)

    def decode_formatted(
        self, ids: list[int], style: str = "llvm", fallback_to_raw: bool = True
    ) -> str:
        """Decode token IDs and run clang-format for exact C++ formatting.

        Args:
            ids: Token IDs to decode
            style: clang-format style (llvm, google, chromium, mozilla, webkit)
            fallback_to_raw: If clang-format fails, return raw decode output

        Returns:
            Formatted C++ source code
        """
        import subprocess

        raw = self.decode(ids)
        try:
            result = subprocess.run(
                ["clang-format", f"--style={style}", "--assume-filename=code.cpp"],
                input=raw,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return raw if fallback_to_raw else raw


def format_cpp(code: str, style: str = "llvm") -> str:
    """Run clang-format on a C++ code string. Standalone utility function.

    Args:
        code: C++ source code string
        style: clang-format style (llvm, google, chromium, mozilla, webkit)

    Returns:
        Formatted code, or original if clang-format unavailable/fails
    """
    import subprocess

    try:
        result = subprocess.run(
            ["clang-format", f"--style={style}", "--assume-filename=code.cpp"],
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return code
