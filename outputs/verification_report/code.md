# CODE examples — formatted code + per-channel sidecar JSON


---

## Example 1 — `WindowsAppSDK.parquet` row 0

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
```
### Roundtrip (our cpp_tokenizer: detok -> retok)
| metric | value |
|---|---|
| text_roundtrip (byte-exact decode) | False |
| reencode_idempotent (load-bearing) | False |
| id_exact (literal stored-ids match) | False |
| id_match_modulo_ws_collapse | False |
| first_id_divergence | 782 |
| n_stored_ids / n_reencoded | 1024 / 1005 |

> id_exact=False is EXPECTED: stored ids preserve raw indentation as repeated literal-space tokens, while encode() canonicalizes whitespace runs to a single <SPACE> sentinel. reencode_idempotent is the load-bearing guarantee (deterministic, self-consistent tokenizer).

### clang-format'd CODE portion
clang-format ok: **True**
```cpp
osoft::TokuchoNamespace::PurojekutoTenpuretNoFeaturePrefix::
    PurojekutoTenpuretNoFeaturePrefixManager
    PurojekutoTenpuretNoFeaturePrefixManager::Open() {
  throw hresult_not_implemented();
}
void Puro jekutoTenpuretNoFeaturePrefixManager
    : PurojekutoTenpuretNoFeaturePrefixManagerT<
          PurojekutoTenpuretNoFeaturePrefixManager> {
  PurojekutoTenpuretNoFeaturePrefixManager() = default;
  static winrt::Microsoft::TokuchoNamespace::PurojekutoTenpuretNoFeaturePrefix::
      PurojekutoTenpuretNoFeaturePrefixManager
      Open();
  void TODO_ReplaceMeWithRealContent();
};
}
namespace win<BOS> // language: primary=c++ standard=c++17 confidence=high
    // <BOS>
    // platform: x86_64-linux-gnu
    // compiler: g++
    // standard: c++17
    // arch: x86_64
    // mode: user
    : PackageDependencyManager::ExistsPackageDependency(
          PSID user, _In_ PCWSTR packageDependencyId) {
  auto lock{std::unique_lock<std::recursive_mutex>(g_lock)};
  // Find it (if we can)
  auto packageDependency{GetPackageDependency(packageDependencyId)};
  if (packageDependency) {
    auto packageDependencyUser{packageDependency->User()};
    if (user) {
      // We're expecting a definition for the specifiedd user            return
      // packageDependencyUser && !!EqualSid(user, packageDependencyUser); }
      // else        {            // We're not expecting a user (i.e. it's for
      // System)            return !packageDependencyUser;        }    } return
      // false;}void MddCore::PackageDependencyMan<BOS>// language: primary=c++
      // standard=c++17 confidence=high
      // <BOS>
      // platform: x86_64-linux-gnu
      // compiler: g++
      // standard: c++17
      // arch: x86_64
      // mode: user
      eVector::const_iterator{};
    }
    return std::ranges::find(m_map->cbegin(), m_map->cend(), key,
                             &FileTypeChoiceVector::value_type::first);
  }
  winrt::Windows::Foundation::Collections::IVector<hstring>
  OrderedMapView::Lookup(hstring const &key) const {
    if (!m_map) {
    throw winrt::h
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```


---

## Example 2 — `WindowsAppSDK.parquet` row 1

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
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
TenpuretNoFeaturePrefixManager::TODO_ReplaceMeWithRealContent() {
  throw hresult_not_implemented();
}
}
jekutoTenpuretNoFeaturePrefixManager
    : PurojekutoTenpuretNoFeaturePrefixManagerT<
          PurojekutoTenpuretNoFeaturePrefixManager> {
  PurojekutoTenpuretNoFeaturePrefixManager() = default;
  static winrt::Microsoft::TokuchoNamespace::PurojekutoTenpuretNoFeaturePrefix::
      PurojekutoTenpuretNoFeaturePrefixManager
      Open();
  void TODO_ReplaceMeWithRealContent();
};
}
namespace win<BOS>// language: primary=c++ standard=c++17 confidence=high
// <BOS>
// platform: x86_64-linux-gnu
// compiler: g++
// standard: c++17
// arch: x86_64
// mode: user
Str, _Outptr_ DEFSTRINGRESULT** result)
{
  *result = nullptr;
  DEFSTRINGRESULT *pSelf = nullptr;
  HRESULT hr = _DefStringResult_Alloc(&pSelf);
  if (SUCCEEDED(hr)) {
    hr = DefStringResult_InitRef(pSelf, pStr);
    if (SUCCEEDED(hr)) {
      *result = pSelf;
      pSelf = nullptr;
    }
  }
  DefStringResult_Delete(pSelf);
  return hr;
}
HRESULT
DefStringResult_NewBuf(__in_opt PCWSTR pStr, _Outptr_ DEFS<BOS>// language: primary=c++ standard=c++17 confidence=high
// <BOS>
// platform: x86_64-linux-gnu
// compiler: g++
// standard: c++17
// arch: x86_64
// mode: user
 RETURN_IF_FAILED(FileDataItemsSection::CreateInstance(this, &u.pDataItems));
 m_sectionType = SectionTypeDataItems;
}
else if (m_sectionType != SectionTypeDataItems) {
  u.pDataItems = nullptr;
  return HRESULT_FROM_WIN32(ERROR_MRM_INVALID_PRI_FILE);
}
*result = u.pDataItems;
return S_OK;
}
HRESULT GetReverseFileMapSection(_Out_ ReverseFileMap **result) {
  *result = nullptr;
  if (m_sectionType == SectionTypeUnknown) {
  R
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```


---

## Example 3 — `WindowsAppSDK.parquet` row 2

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
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
cslen(s) + 1) * sizeof(*s));
}

private:
static std::recursive_mutex s_lock;
static MddCore::PackageGraph
    s<BOS> // language: primary=c++ standard=c++17 confidence=high
           // <BOS>
           // platform: x86_64-linux-gnu
           // compiler: g++
           // standard: c++17
           // arch: x86_64
           // mode: user
    delete[] m_pDecisions;
m_pDecisions = NULL;
}
}
HRESULT
TestDecisionCollection::InitFromTest rs(__in PCWSTR pPrefix,
                                        __in TestQualifierSetCollection *pSets,
                                        __in bool required);
HRESULT InitFromTestVars(__in PCWSTR pPrefix,
                         __in TestQualifierSetCollection *pSets) {
  return InitFromTestVars(pPrefix, pSets, true);
}
int GetNumTestDecisions() const { return m_numDecisions; }
HRESULT GetTestDecision(__in int index, __out TestDecision *pDecisionOut) const;
HRESULT GetTestDecision(__in PCWSTR pId,
                        __out TestDecision *pDecisionOut) const;
bool TryGetDecision(__in PCWSTR pId, __in const IDecisionInfo *pDecisions,
                    __in const UnifiedEnvironment *pEnvironment,
                    __inout DecisionResult *pDecisionOut) const;
static bool TryGetDecision(__in const TestDecision *pTestSpec,
                           __in const IDecisionInfo *pDecisions,
                           __in const UnifiedEnvironment *pEnvironment,
                           __inout DecisionResult *pDecisionOut);
HRESULT GetOrAddDecision(__in PCWSTR pId, __in DecisionInfoBuilder *pDecisions,
                         __inout_opt DecisionResult *pDecisionOut);
static HRESULT GetOrAddDecision(__in const TestDecision *pTestSpec,
                                __in DecisionInfoBuilder *pDecisions,
                                __inout_opt DecisionResult *pDecisionOut);

protected:
int m_numDecisions;
TestDataArray<String> m_testStrings;
TestStringArray *m_pSpecs;
TestDecision *m_pDecisions;
}
;
class TestDecisionInfo {
public:
  TestDecisionInfo();
  ~TestDecisionInfo();
  HRESULT InitDataFromTestVars(__in PCWSTR pPrefix);
 HRESULT ApplyTestData(__inout DecisionInfoB
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```


---

## Example 4 — `WindowsAppSDK.parquet` row 3

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
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
eInfo FromPackageInfoReference(
    PACKAGE_INFO_REFERENCE packageInfoReference,
    const UINT32 flags = PACKAGE_FILTER_HEAD | PACKAGE_FILTER_DIRECT |
                         PACKAGE_FILTER_OPTIONAL | PACKAGE_FILTER_RESOURCE |
                         PACKAGE_FILTER_BUNDLE,
    const PackagePathType packagePathType = PackagePathType_Effective) {
  UINT32 bufferLength{};
  const LONG rc{appmodel::GetPackageInfo2(packageInfoReference, flags,
                                          packagePathType, &bufferLength,
                                          nullptr, nullptr)};
  THROW_HR_IF(HRESULT_FROM_WIN32(rc), rc != ERROR_INSUFFICIENT_BUFFER);
  std::unique_ptr<BYTE[]> buffer{std::make_unique<BYTE[]>(bufferLength)};
  UINT32 count{};
  THROW_IF_WIN32_ERROR(appmodel::GetPackageInfo2(packageInfoReference, flags,
                                                 packagePathType, &bufferLength,
                                                 buffer.get(), &count));
  auto packageInfo{PackageInfo(buffer.get(), count)};
  buffer.release();
  return packageInfo;
}
 PackageInfo() = defaul<BOS>// language: primary=c++ standard=c++17 confidence=high
// <BOS>
// platform: x86_64-linux-gnu
// compiler: g++
// standard: c++17
// arch: x86_64
// mode: user
== medStr);
 }
 void DefStringResultTests_Reference::GetWritableRef(void) {
   PWSTR pRef;
   size_t cchRef;
   PWSTR stuff = L"pqowl"; // must be shorter than medStr
 size_t cchStuff = _countof(L"pqowce>, public DefStringResult_Base{public:    TEST_CLASS(DefStringResultTests_Reference);    TEST_METHOD_SETUP(Setup);    TEST_METHOD_CLEANUP(Cleanup);    TEST_METHOD(New);    TEST_METHOD(Basics);    TEST_METHOD(InitRef);    TEST_METHOD(InitRef_Junk);    TEST_METHOD(GetRef);    TEST_METHOD(GetWritableRef);    TEST_METHOD(SetRef);    TEST_METHOD(SetBuf);    TEST_METHOD(SetBufNull);    TEST_METHOD(AcquireBuf);    TEST_METHOD(AcquireBufNull);    TEST_METHOD(SetContents);    TEST_METHOD(SetEmptyContents);    TEST_METHOD(ReleaseAfterAcquire);    TEST_METHOD(ReleaseEmptyBuf);    TEST_METHOD(Length);    TEST_METHOD(Compare);    TEST_METHOD(Concat);    TEST_METHOD(Copy);    TEST_METHOD(InvalidTests);};bool DefStringResultTests_Reference::Setup(void) { return SetupBase(DefResultType_Reference); }bool DefStringRes
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```


---

## Example 5 — `WindowsAppSDK.parquet` row 4

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
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
:
 m_packageInfoBuffer(std::move(other.m_packageInfoBuffer)),
 m_count(other.m_count)
 {
   m_packageInfo = reinterpret_cast<PACKAGE_INFO *>(m_packageInfoBuffer.get());
   other.m_packageInfo = nullptr;
 }
 PackageInfo(void* buffer, size_t <BOS>// language: primary=c++ standard=c++17 confidence=high
// <BOS>
// platform: x86_64-linux-gnu
// compiler: g++
// standard: c++17
// arch: x86_64
// mode: user
 {
  return 0;
 }
 auto base{ (((s[0] == L'0') && s[1] == L'x') ? 16 : 10) };
 return wcstoul(s, nullptr, base);
 }
 static uint64_t uint64_from_string(PCWSTR s) {
   if (!s)
     <BOS> // language: primary=c++ standard=c++17 confidence=high
           // <BOS>
           // platform: x86_64-linux-gnu
           // compiler: g++
           // standard: c++17
           // arch: x86_64
           // mode: user
         decimal truncate() const {
       decimal value{};
       THROW_IF_FAILED(
           ::VarDecFix(const_cast<DECIMAL *>(&m_decimal), &value.m_decimal));
       return value;
     }
   /// Return the integral digits rounded dow<BOS>// language: primary=c++
   /// standard=c++17 confidence=high
   // <BOS>
   // platform: x86_64-linux-gnu
   // compiler: g++
   // standard: c++17
   // arch: x86_64
   // mode: user
   RESULT CreateInstance(_In_ AtomPoolGroup * pAtoms, _In_ int major,
                         _In_ int minor,
                         _Outptr_ MrmEnvironment **environment) {
     return MrmEnvironment::CreateInstance(
         pAtoms, &CoreEnvironmentInitializer, major, minor,
         e<BOS> // language: primary=c++ standard=c++17 confidence=high
             // <BOS>
             // platform: x86_64-linux-gnu
             // compiler: g++
             // standard: c++17
             // arch: x86_64
             // mode: user
             ativeToScope,
         _Inout_ int scopeIndex, _Inout_ StringResult *pNameOut) const {
       return m_pNames->TryGetRelativeScopeName(relativeToScope, scopeIndex,
                                                pNameOut);
     }
 HRESULT GetNumDescendents(_In_ int
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```


---

## Example 6 — `WindowsAppSDK.parquet` row 5

### Provenance
```json
{
  "repo": "",
  "filepath": "",
  "commit_hash": "",
  "timestamp": "",
  "pr_number": null,
  "sha_in_doc": null,
  "repo_in_doc": null,
  "brief": null
}
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
if (this != &other) {
  m_packageInfoBuffer = std::move(other.m_packageInfoBuffer);
  m_packageInfo = reinterpret_cast<PACKAGE_INFO *>(m_packageInfoBuffer.get());
  m_count = other.m_count;
  other.m_packageInfo = nullptr;
  other.m_count = 0;
}
return *this;
}
void Reset() {
  m_packageInfo = nullptr;
  <BOS> // language: primary=c++ standard=c++17 confidence=high
      // <BOS>
      // platform: x86_64-linux-gnu
      // compiler: g++
      // standard: c++17
      // arch: x86_64
      // mode: user
      tificateEkuValidator::IsPackageValid(
          winrt::Windows::Foundation::IInspectable const &appxPackagingObject) {
    winrt::com_ptr<IAppxPackageReader> packageReader;
    if (SUCCEEDED(appxPackagingObject.as(IID_PPV_ARGS(&packageReader)))) {
      winrt::com_ptr<IAppxFile> signatureFile;
      if (FAILED_LOG(packageReader->GetFootprintFile(
              APPX_FOOTPRINT_FILE_TYPE_SIGNATURE, signatureFile.put()))) {
        return false;
      }
      return CheckSignature(signatureFile.get());
    }
    winrt::com_ptr<IAppxBundleReader> bundleReader;
    if (SUCCEEDED(appxPackagingObject.as(IID_PPV_ARGS(&bundleReader)))) {
      winrt::com_ptr<IAppxFile> signatureFile;
      if (FAILED_LOG(bundleReader->GetFootprintFile(
              APPX_BUNDLE_FOOTPRINT_FILE_TYPE_SIGNATURE,
              signatureFile.put()))) {
        return false;
      }
      return CheckSignature(signatureFile.get());
    }
    THROW_HR(APPX_E_CORRUPT_CONTENT);
  }
  bool PackageCertificateEkuValidator : kageCertificateEkuValidator
      : PackageCertificateEkuValidatorT<PackageCertificateEkuValidator> {
    PackageCertificateEkuValidator(hstring const &expectedCertificateEku);
    bool IsPackageValid(
        winrt::Windows::Foundation::IInspectable const &appxPackagingObject);

  private:
    bool CheckSignature(IAppxFile * signatureFile);
    hstring m_expectedEku{};
  };
}
namespace winr
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
    "start": 0,
    "count": 40
  },
  "per_token": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 12,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 13,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 14,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 15,
      "tok_id": 7248,
      "tok": "s",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 16,
      "tok_id": 2020,
      "tok": "tan",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 17,
      "tok_id": 7234,
      "tok": "d",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 18,
      "tok_id": 7591,
      "tok": "ard",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 19,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 20,
      "tok_id": 7233,
      "tok": "c",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 21,
      "tok_id": 335,
      "tok": "++",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 22,
      "tok_id": 4917,
      "tok": "17",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 23,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 24,
      "tok_id": 4725,
      "tok": "co",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 25,
      "tok_id": 7243,
      "tok": "n",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 26,
      "tok_id": 7322,
      "tok": "fi",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 27,
      "tok_id": 4704,
      "tok": "de",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 28,
      "tok_id": 39482,
      "tok": "nce",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 29,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 30,
      "tok_id": 9123,
      "tok": "high",
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 31,
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
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 32,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
     
```
