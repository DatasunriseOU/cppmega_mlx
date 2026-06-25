# COMMIT examples — full atomic block (docstring + PRE + POST + diff) + D-family edit channels


---

## Example 1 — `s2n-tls_r2500.parquet` row 1

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "crypto/s2n_certificate.c",
  "commit_hash": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "timestamp": "2019-05-28T13:51:00-07:00",
  "pr_number": null,
  "sha_in_doc": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Support wildcard matching to select server certs"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Support wildcard matching to select server certs
 *
 * @repo aws/s2n-tls
 * File: crypto/s2n_certificate.c
 * @sha 032d8f3d707c75e98a1a211315f1173ec93fa918
 *
 * @details
 * Only certificates with "simple" wildcards are supported, where simple
 * means a single * in the left-most DNS label. If s2n is configured with
 * a certificate that matches a domain name exactly *and* a simple wildcard
 * certificate that matches a domain name, we will prefer the exact match.
 * 
 * TODO:
 * - [] Add more test cases for wildcardify-ing
 * - [] Update fuzz test to cover wildcard
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Support wildcard matching to select server certs
 *
 * @repo aws/s2n-tls
 * File: crypto/s2n_certificate.c
 * @sha 032d8f3d707c75e98a1a211315f1173ec93fa918
 *
 * @details
 * Only certificates with "simple" wildcards are supported, where simple
 * means a single * in the left-most DNS label. If s2n is configured with
 * a certificate that matches a domain name exactly *and* a simple wildcard
 * certificate that matches a domain name, we will prefer the exact match.
 *
 * TODO:
 * - [] Add more test cases for wildcardify-ing
 * - [] Update fuzz test to cover wildcard
 */
// === CONTEXT ===
int s2n_cert_public_key_set_rsa_from_openssl(s2n_cert_public_key *public_key,
                                             RSA *openssl_rsa) {
  notnull_check(openssl_rsa);
  notnull_check(public_key);
  public_key->key.rsa_key.rsa = openssl_rsa;
  return 0;
}
// === DIFF ===
diff-- git a / crypto / s2n_certificate.c b / crypto /
        s2n_certificate.c index c47c56384..05206ff3a 100644 -- -a / crypto /
        s2n_certificate.c++ +
    b / crypto / s2n_certificate.c @ @-33,
    7 + 33,
    6 @ @ static const s2n_authentication_method cert_type_to_auth_method[] = {
        [S2N_CERT_TYPE_ECDSA_SIGN] = S2N_AUTHENTICATION_ECDSA,
};

-
    int
    s2n_cert_public_key_set_rsa_from_openssl(s2n_cert_public_key *public_key,
                                             RSA *openssl_rsa) {
  notnull_check(openssl_rsa);
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/crypto/s2n_certificate.c b/crypto/s2n_certificate.c
index c47c56384..05206ff3a 100644
--- a/crypto/s2n_certificate.c
+++ b/crypto/s2n_certificate.c
@@ -33,7 +33,6 @@ static const s2n_authentication_method cert_type_to_auth_method[] = {
 [S2N_CERT_TYPE_ECDSA_SIGN] = S2N_AUTHENTICATION_ECDSA,
 };
 
-
 int s2n_cert_public_key_set_rsa_from_openssl(s2n_cert_public_key *public_key, RSA *openssl_rsa)
 {
 notnull_check(openssl_rsa);
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 486,
    "count": 40
  },
  "per_token": [
    {
      "i": 486,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 487,
      "tok_id": 327,
      "tok": "==",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 488,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 489,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 490,
      "tok_id": 133,
      "tok": "int",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 491,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 492,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 493,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 494,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 495,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 496,
      "tok_id": 7233,
      "tok": "c",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 497,
      "tok_id": 4760,
      "tok": "er",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 498,
      "tok_id": 7249,
      "tok": "t",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 499,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 500,
      "tok_id": 90,
      "tok": "public",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 501,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 502,
      "tok_id": 7508,
      "tok": "key",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 503,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 504,
      "tok_id": 1504,
      "tok": "set",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 505,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 506,
      "tok_id": 13783,
      "tok": "rsa",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 507,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 508,
      "tok_id": 7556,
      "tok": "from",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 509,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 510,
      "tok_id": 4778,
      "tok": "open",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 511,
      "tok_id": 7368,
      "tok": "ss",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 512,
      "tok_id": 923,
      "tok": "l",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 513,
      "tok_id": 352,
      "tok": "(",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 6297,
      "B_ast_node": 1,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 514,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 515,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 516,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 829996912,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 744721255,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    
```


---

## Example 2 — `s2n-tls_r2500.parquet` row 3

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/cleanup.h",
  "commit_hash": "3e91e4531bad3bf9173c04a768856504931382b5",
  "timestamp": "2019-05-20T11:52:46-07:00",
  "pr_number": null,
  "sha_in_doc": "3e91e4531bad3bf9173c04a768856504931382b5",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory
 * length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
// === CONTEXT ===
_INLINE_ void compressed_idx_dv_ar_cleanup(IN OUT compressed_idx_dv_ar_t *o) {
  for (int i = 0; i < N0; i++) {
    secure_clean((uint8_t *)&o[i], sizeof(*o[0]));
  }
}
// === DIFF ===
diff-- git a / pq - crypto / bike / cleanup.h b / pq -
    crypto / bike / cleanup.h index bb2b676ec..675681d17 100644 -- -a / pq -
    crypto / bike / cleanup.h++ + b / pq - crypto / bike / cleanup.h @ @-47,
    6 + 47,
    6 @ @_INLINE_ void
    compressed_idx_dv_ar_cleanup(IN OUT compressed_idx_dv_ar_t *o) {
  for (int i = 0; i < N0; i++) {
    -secure_clean((uint8_t *)&o[i], sizeof(*o[0]));
    +secure_clean((uint8_t *)&(*o)[i], sizeof((*o)[0]));
  }
}
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/pq-crypto/bike/cleanup.h b/pq-crypto/bike/cleanup.h
index bb2b676ec..675681d17 100644
--- a/pq-crypto/bike/cleanup.h
+++ b/pq-crypto/bike/cleanup.h
@@ -47,6 +47,6 @@ _INLINE_ void compressed_idx_dv_ar_cleanup(IN OUT compressed_idx_dv_ar_t *o)
 {
 for(int i=0; i < N0; i++)
 {
- secure_clean((uint8_t*)&o[i], sizeof(*o[0])); 
+ secure_clean((uint8_t*)&(*o)[i], sizeof((*o)[0]));
 }
 }
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 236,
    "count": 40
  },
  "per_token": [
    {
      "i": 236,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 237,
      "tok_id": 327,
      "tok": "==",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 238,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 239,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 240,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 241,
      "tok_id": 7297,
      "tok": "IN",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 242,
      "tok_id": 924,
      "tok": "L",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 243,
      "tok_id": 7765,
      "tok": "INE",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 244,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 245,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 246,
      "tok_id": 125,
      "tok": "void",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 247,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 248,
      "tok_id": 4725,
      "tok": "co",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 249,
      "tok_id": 7242,
      "tok": "m",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 250,
      "tok_id": 4700,
      "tok": "pre",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 251,
      "tok_id": 7368,
      "tok": "ss",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 252,
      "tok_id": 4764,
      "tok": "ed",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 253,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 254,
      "tok_id": 4841,
      "tok": "idx",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 255,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 256,
      "tok_id": 13500,
      "tok": "dv",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 257,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 258,
      "tok_id": 7262,
      "tok": "ar",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 259,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 260,
      "tok_id": 7233,
      "tok": "c",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 261,
      "tok_id": 923,
      "tok": "l",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 262,
      "tok_id": 7235,
      "tok": "e",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 263,
      "tok_id": 7263,
      "tok": "an",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 264,
      "tok_id": 4729,
      "tok": "up",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 265,
      "tok_id": 352,
      "tok": "(",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 43,
      "B_ast_node": 1,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 266,
      "tok_id": 7297,
      "tok": "IN",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 967917155,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 267,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
   
```


---

## Example 3 — `s2n-tls_r2500.parquet` row 5

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/utilities.c",
  "commit_hash": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "timestamp": "2019-05-17T21:13:33-07:00",
  "pr_number": null,
  "sha_in_doc": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Use inttypes macro to print uint64_t"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
// === CONTEXT ===
_INLINE_ void print_uint64(IN const uint64_t val) {
  // If printing in BE is requried swap the order of bytes
#ifdef PRINT_IN_BE
  uint64_t tmp = bswap_64(val);
#else
  uint64_t tmp = val;
#endif
#ifdef WIN32
  printf("%.16I64x", tmp);
#else
  printf("%.16lx", tmp);
#endif
#ifndef NO_SPACE
  printf(" ");
#endif
}
// === DIFF ===
diff-- git a / pq - crypto / bike / utilities.c b / pq -
    crypto / bike / utilities.c index 15e7268ac..d1828fc26 100644 -- -a / pq -
    crypto / bike / utilities.c++ + b / pq - crypto / bike / utilities.c @ @-9,
    6 + 9, 8 @ @ *The license is detailed in the file LICENSE.md,
    and applies to this file.***************************************************
                ************************* /

            +#include<inttypes.h> +
#include "utilities.h"

#define BITS_IN_QW 64ULL
        @ @-38,
    7 + 40,
    7 @ @_INLINE_ void print_uint64(IN const uint64_t val)
#ifdef WIN32
        printf("%.16I64x", tmp);
#else
        - printf("%.16lx", tmp);
+ printf("%.16" PRIu64, tmp);
#endif

#ifndef NO_SPACE
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/pq-crypto/bike/utilities.c b/pq-crypto/bike/utilities.c
index 15e7268ac..d1828fc26 100644
--- a/pq-crypto/bike/utilities.c
+++ b/pq-crypto/bike/utilities.c
@@ -9,6 +9,8 @@
 * The license is detailed in the file LICENSE.md, and applies to this file.
 * ***************************************************************************/
 
+#include <inttypes.h>
+
 #include "utilities.h"
 
 #define BITS_IN_QW 64ULL
@@ -38,7 +40,7 @@ _INLINE_ void print_uint64(IN const uint64_t val)
 #ifdef WIN32
 printf("%.16I64x", tmp);
 #else
- printf("%.16lx", tmp);
+ printf("%.16"PRIu64, tmp);
 #endif
 
 #ifndef NO_SPACE
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 211,
    "count": 40
  },
  "per_token": [
    {
      "i": 211,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 212,
      "tok_id": 327,
      "tok": "==",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 213,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 214,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 215,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 216,
      "tok_id": 7297,
      "tok": "IN",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 217,
      "tok_id": 924,
      "tok": "L",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 218,
      "tok_id": 7765,
      "tok": "INE",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 219,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 220,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 221,
      "tok_id": 125,
      "tok": "void",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 222,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 223,
      "tok_id": 1861,
      "tok": "print",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 224,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 225,
      "tok_id": 921,
      "tok": "u",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 226,
      "tok_id": 133,
      "tok": "int",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 227,
      "tok_id": 4964,
      "tok": "64",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 228,
      "tok_id": 352,
      "tok": "(",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 229,
      "tok_id": 7297,
      "tok": "IN",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 230,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 231,
      "tok_id": 65,
      "tok": "const",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 232,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 233,
      "tok_id": 146,
      "tok": "uint64_t",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 234,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 235,
      "tok_id": 7250,
      "tok": "v",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 236,
      "tok_id": 4767,
      "tok": "al",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 237,
      "tok_id": 353,
      "tok": ")",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 238,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1,
      "B_ast_node": 1,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 239,
      "tok_id": 350,
      "tok": "{",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 1,
      "B_ast_node": 10,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 240,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 1,
      "B_ast_node": 10,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 241,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 1,
      "B_ast_node": 10,
      "C_symbol": 1125301607,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 242,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
   
```


---

## Example 4 — `s2n-tls_r2500.parquet` row 9

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "tls/s2n_cipher_preferences.h",
  "commit_hash": "44f4487ffa1685b78e7af9ea6aea34e9a9f8833a",
  "timestamp": "2019-05-07T15:11:00-07:00",
  "pr_number": null,
  "sha_in_doc": "44f4487ffa1685b78e7af9ea6aea34e9a9f8833a",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Add configuration to s2n_cipher_preferences to determine when clients send key exchange extensions for ECHDE and SIKE"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Add configuration to s2n_cipher_preferences to determine when clients send key exchange extensions for ECHDE and SIKE
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_cipher_preferences.h
 * @sha 44f4487ffa1685b78e7af9ea6aea34e9a9f8833a
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Add configuration to s2n_cipher_preferences to determine when clients
 * send key exchange extensions for ECHDE and SIKE
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_cipher_preferences.h
 * @sha 44f4487ffa1685b78e7af9ea6aea34e9a9f8833a
 */
// === CONTEXT ===
struct s2n_cipher_preferences {
  uint8_t count;
  struct s2n_cipher_suite **suites;
  int minimum_protocol_version;
}
        // === DIFF ===
        diff-- git a /
        tls / s2n_cipher_preferences.h b / tls /
        s2n_cipher_preferences.h index a273a83bc..eb83cfd63 100644 -- -a / tls /
        s2n_cipher_preferences.h++ +
    b / tls / s2n_cipher_preferences.h @ @-19,
    10 + 19,
    15 @ @

#include "tls/s2n_cipher_suites.h"

        + +#define S2N_ECC_EXTENSION_ENABLED 0x01 +
        #define S2N_SIKE_EXTENSION_ENABLED 0x02 +
        struct s2n_cipher_preferences {
  uint8_t count;
  struct s2n_cipher_suite **suites;
  int minimum_protocol_version;
  + uint8_t extension_flag;
};

extern const struct s2n_cipher_preferences cipher_preferences_20140601;
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/tls/s2n_cipher_preferences.h b/tls/s2n_cipher_preferences.h
index a273a83bc..eb83cfd63 100644
--- a/tls/s2n_cipher_preferences.h
+++ b/tls/s2n_cipher_preferences.h
@@ -19,10 +19,15 @@
 
 #include "tls/s2n_cipher_suites.h"
 
+
+#define S2N_ECC_EXTENSION_ENABLED 0x01
+#define S2N_SIKE_EXTENSION_ENABLED 0x02
+
 struct s2n_cipher_preferences {
 uint8_t count;
 struct s2n_cipher_suite **suites;
 int minimum_protocol_version;
+ uint8_t extension_flag;
 };
 
 extern const struct s2n_cipher_preferences cipher_preferences_20140601;
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 258,
    "count": 40
  },
  "per_token": [
    {
      "i": 258,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 259,
      "tok_id": 327,
      "tok": "==",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 260,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 261,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 262,
      "tok_id": 83,
      "tok": "struct",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 263,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 264,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 265,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 266,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 267,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 268,
      "tok_id": 25126,
      "tok": "ciph",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 269,
      "tok_id": 4760,
      "tok": "er",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 270,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 271,
      "tok_id": 4700,
      "tok": "pre",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 272,
      "tok_id": 7236,
      "tok": "f",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 273,
      "tok_id": 4760,
      "tok": "er",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 274,
      "tok_id": 4750,
      "tok": "ence",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 275,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 276,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 277,
      "tok_id": 350,
      "tok": "{",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 278,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 279,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 280,
      "tok_id": 143,
      "tok": "uint8_t",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 281,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 282,
      "tok_id": 4853,
      "tok": "count",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 283,
      "tok_id": 358,
      "tok": ";",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 284,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 285,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 1095,
      "B_ast_node": 3,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 286,
      "tok_id": 83,
      "tok": "struct",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 1,
      "B_ast_node": 7,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 287,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 1,
      "B_ast_node": 7,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 288,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 1102691786,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 289,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
     
```


---

## Example 5 — `s2n-tls_r2500.parquet` row 12

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "tls/s2n_kem.c",
  "commit_hash": "62c79fb2e72fae9d70995abeb5b74cd6828e3b41",
  "timestamp": "2019-04-23T11:03:23-07:00",
  "pr_number": null,
  "sha_in_doc": "62c79fb2e72fae9d70995abeb5b74cd6828e3b41",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Add sike fuzz tests"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Add sike fuzz tests
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_kem.c
 * @sha 62c79fb2e72fae9d70995abeb5b74cd6828e3b41
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Add sike fuzz tests
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_kem.c
 * @sha 62c79fb2e72fae9d70995abeb5b74cd6828e3b41
 */
// === CONTEXT ===
int s2n_kem_generate_keypair(struct s2n_kem_keypair *kem_keys) {
  notnull_check(kem_keys);
  const struct s2n_kem *kem = kem_keys->negotiated_kem;
  notnull_check(kem->generate_keypair);
  eq_check(kem_keys->public_key.size, kem->public_key_length);
  notnull_check(kem_keys->public_key.data);
  /* The private key is needed for client_key_recv and must be saved */
  GUARD(s2n_alloc(&kem_keys->private_key, kem->private_key_length));
  GUARD(kem->generate_keypair(kem_keys->public_key.data,
                              kem_keys->private_key.data));
  return 0;
}
// === DIFF ===
diff-- git a / tls / s2n_kem.c b / tls /
        s2n_kem.c index 8f7ebf723..473207ea2 100644 -- -a / tls / s2n_kem.c++ +
    b / tls / s2n_kem.c @ @-13,
    6 + 13,
    8 @ @ * permissions and limitations under the License.* /

                                    +#include "pq-crypto/sike/sike_p503_kem.h" +
#include "stuffer/s2n_stuffer.h"

#include "tls/s2n_kem.h"
                                @ @-20,
    6 + 22,
    16 @ @
#include "utils/s2n_mem.h"
#include "utils/s2n_safety.h"

    +const struct s2n_kem s2n_sike_r1_p503 = {
        +.public_key_length = SIKE_P503_PUBLIC_KEY_BYTES,
        +.private_key_length = SIKE_P503_SECRET_KEY_BYTES,
        +.shared_secret_key_length = SIKE_P503_SHARED_SECRET_BYTES,
        +.ciphertext_length = SIKE_P503_CIPHERTEXT_BYTES,
        +.generate_keypair = &SIKE_P503_crypto_kem_keypair,
        +.encapsulate = &SIKE_P503_crypto_kem_enc,
        +.decapsulate = &SIKE_P503_crypto_kem_dec,
        +};
+ int s2n_kem_generate_keypair(struct s2n_kem_keypair *kem_keys) {
  notnull_check(kem_keys);
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/tls/s2n_kem.c b/tls/s2n_kem.c
index 8f7ebf723..473207ea2 100644
--- a/tls/s2n_kem.c
+++ b/tls/s2n_kem.c
@@ -13,6 +13,8 @@
 * permissions and limitations under the License.
 */
 
+#include "pq-crypto/sike/sike_p503_kem.h"
+
 #include "stuffer/s2n_stuffer.h"
 
 #include "tls/s2n_kem.h"
@@ -20,6 +22,16 @@
 #include "utils/s2n_safety.h"
 #include "utils/s2n_mem.h"
 
+const struct s2n_kem s2n_sike_r1_p503 = {
+ .public_key_length = SIKE_P503_PUBLIC_KEY_BYTES,
+ .private_key_length = SIKE_P503_SECRET_KEY_BYTES,
+ .shared_secret_key_length = SIKE_P503_SHARED_SECRET_BYTES,
+ .ciphertext_length = SIKE_P503_CIPHERTEXT_BYTES,
+ .generate_keypair = &SIKE_P503_crypto_kem_keypair,
+ .encapsulate = &SIKE_P503_crypto_kem_enc,
+ .decapsulate = &SIKE_P503_crypto_kem_dec,
+};
+
 int s2n_kem_generate_keypair(struct s2n_kem_keypair *kem_keys)
 {
 notnull_check(kem_keys);
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 201,
    "count": 40
  },
  "per_token": [
    {
      "i": 201,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 202,
      "tok_id": 327,
      "tok": "==",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 203,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 204,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 205,
      "tok_id": 133,
      "tok": "int",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 206,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 207,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1821124260,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 208,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1821124260,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 209,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1821124260,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 210,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1821124260,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 211,
      "tok_id": 29390,
      "tok": "kem",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1821124260,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 212,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 213,
      "tok_id": 1647,
      "tok": "generate",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 1038756647,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 214,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 1038756647,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 215,
      "tok_id": 7508,
      "tok": "key",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 1038756647,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 216,
      "tok_id": 1505,
      "tok": "pair",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 1038756647,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 217,
      "tok_id": 352,
      "tok": "(",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 3351,
      "B_ast_node": 1,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 218,
      "tok_id": 83,
      "tok": "struct",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 219,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 220,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 221,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 222,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 223,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 224,
      "tok_id": 29390,
      "tok": "kem",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 225,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 226,
      "tok_id": 7508,
      "tok": "key",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 227,
      "tok_id": 1505,
      "tok": "pair",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 1937251339,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 228,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 229,
      "tok_id": 364,
      "tok": "*",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 230,
      "tok_id": 29390,
      "tok": "kem",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 231,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "3:decl/signature",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 6,
      "C_symbol": 597959663,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 1,
  
```


---

## Example 6 — `s2n-tls_r2500.parquet` row 14

### Provenance
```json
{
  "repo": "aws/s2n-tls",
  "filepath": "tls/s2n_kem.h",
  "commit_hash": "62c79fb2e72fae9d70995abeb5b74cd6828e3b41",
  "timestamp": "2019-04-23T11:03:23-07:00",
  "pr_number": null,
  "sha_in_doc": "62c79fb2e72fae9d70995abeb5b74cd6828e3b41",
  "repo_in_doc": "aws/s2n-tls",
  "brief": "Add sike fuzz tests"
}
```
### PR / commit docstring (head of the atomic block)
```c
/**
 * @brief Add sike fuzz tests
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_kem.h
 * @sha 62c79fb2e72fae9d70995abeb5b74cd6828e3b41
 */
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | True |
| reencode_idempotent (load-bearing) | True |
| id_exact (literal stored-ids match) | True |
| id_match_modulo_ws_collapse | True |
| first_id_divergence | None |
| n_stored_ids / n_reencoded | 1024 / 1024 |

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
/**
 * @brief Add sike fuzz tests
 *
 * @repo aws/s2n-tls
 * File: tls/s2n_kem.h
 * @sha 62c79fb2e72fae9d70995abeb5b74cd6828e3b41
 */
// === CONTEXT ===
typedef uint8_t kem_extension_size typedef uint16_t
    kem_public_key_size typedef uint16_t kem_private_key_size typedef uint16_t
        kem_shared_secret_size typedef uint16_t
            kem_ciphertext_key_size struct s2n_kem_keypair {
  const struct s2n_kem *negotiated_kem;
  struct s2n_blob public_key;
  struct s2n_blob private_key;
}
        // === DIFF ===
        diff-- git a /
        tls / s2n_kem.h b / tls /
        s2n_kem.h index baf8b26cb..25a495b2c 100644 -- -a / tls / s2n_kem.h++ +
    b / tls / s2n_kem.h @ @-42,
    6 + 42, 8 @ @ struct s2n_kem_keypair {
  struct s2n_blob private_key;
};

+ extern const struct s2n_kem s2n_sike_r1_p503;
+ extern int s2n_kem_generate_keypair(struct s2n_kem_keypair *kem_keys);

 extern int s2n_kem_encapsulate(const struct s2n_kem_keypair *kem_keys, struct s2n_blob *shared_secret,
```
### Git diff (tail of the atomic block)
```diff
=== DIFF ===
diff --git a/tls/s2n_kem.h b/tls/s2n_kem.h
index baf8b26cb..25a495b2c 100644
--- a/tls/s2n_kem.h
+++ b/tls/s2n_kem.h
@@ -42,6 +42,8 @@ struct s2n_kem_keypair {
 struct s2n_blob private_key;
 };
 
+extern const struct s2n_kem s2n_sike_r1_p503;
+
 extern int s2n_kem_generate_keypair(struct s2n_kem_keypair *kem_keys);
 
 extern int s2n_kem_encapsulate(const struct s2n_kem_keypair *kem_keys, struct s2n_blob *shared_secret,
```
### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)
```json
{
  "_legend": {
    "edit_op": {
      "0": "UNCHANGED",
      "1": "INSERTED",
      "2": "MODIFIED",
      "3": "CONTEXT"
    },
    "structure_id": {
      "0": "none/ws",
      "1": "comment",
      "2": "preproc",
      "3": "decl/signature",
      "4": "body/stmt",
      "5": "expr",
      "6": "identifier-ctx",
      "7": "literal-ctx",
      "8": "misc"
    },
    "def_use": {
      "0": "none",
      "1": "def",
      "2": "use",
      "3": "def+use"
    },
    "families": {
      "A": "platform",
      "B": "structure (syntax+structure)",
      "C": "graph-semantic (symbol/def_use/call/type + edges)",
      "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)"
    }
  },
  "window": {
    "start": 269,
    "count": 40
  },
  "per_token": [
    {
      "i": 269,
      "tok_id": 7508,
      "tok": "key",
      "A_platform": 0,
      "B_structure": "1:comment",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 17,
      "B_ast_node": 9,
      "C_symbol": 0,
      "C_def_use": "2:use",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 270,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "1:comment",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 17,
      "B_ast_node": 9,
      "C_symbol": 0,
      "C_def_use": "2:use",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 271,
      "tok_id": 1550,
      "tok": "size",
      "A_platform": 0,
      "B_structure": "1:comment",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 17,
      "B_ast_node": 9,
      "C_symbol": 0,
      "C_def_use": "2:use",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 272,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 273,
      "tok_id": 83,
      "tok": "struct",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 274,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 275,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 276,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 277,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 278,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 279,
      "tok_id": 29390,
      "tok": "kem",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 280,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 281,
      "tok_id": 7508,
      "tok": "key",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 282,
      "tok_id": 1505,
      "tok": "pair",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 283,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 284,
      "tok_id": 350,
      "tok": "{",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 285,
      "tok_id": 47,
      "tok": "<NL>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 286,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 1,
      "B_sibling": 19,
      "B_ast_node": 3,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 287,
      "tok_id": 65,
      "tok": "const",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 288,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 289,
      "tok_id": 83,
      "tok": "struct",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 290,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 291,
      "tok_id": 7248,
      "tok": "s",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 292,
      "tok_id": 4902,
      "tok": "2",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 293,
      "tok_id": 7243,
      "tok": "n",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 294,
      "tok_id": 377,
      "tok": "_",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 295,
      "tok_id": 29390,
      "tok": "kem",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 3,
      "B_sibling": 0,
      "B_ast_node": 30,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 296,
      "tok_id": 46,
      "tok": "<SPACE>",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 297,
      "tok_id": 364,
      "tok": "*",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 298,
      "tok_id": 18881,
      "tok": "negoti",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 299,
      "tok_id": 4742,
      "tok": "ate",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 300,
      "tok_id": 7234,
      "tok": "d",
      "A_platform": 0,
      "B_structure": "4:body/stmt",
      "B_dep_lvl": 0,
      "B_ast_depth": 2,
      "B_sibling": 0,
      "B_ast_node": 7,
      "C_symbol": 468459033,
      "C_def_use": "1:def
```
