# TileLang runtime-stack promotion

Path C now pins the H200-proven native stack:

- TileLang `a760fe587995def0f3108ee204be453d87467c5d`, which serializes identical JIT cache keys across local ranks.
- Vendored TVM `84af17279edb5edad29749bd6b0eea2ed9393105`, which preserves cross-library exception RTTI while isolating the LLVM no-RTTI shim.
- tvm-ffi `e4353339293459e3e8a393afc1b6a6a869e75b13`, which restores the Python thread state during C++ exception unwinding.

The immutable candidate wheels passed the exact 2xH200 R2 gate (`129b949cdd1348e1b386674a9da6b88d`) and the exact R4 gate (`73564e9432464d44bc18212d939f9f7e`). R4 completed all three TP/SP/CP parity tests with no source mutation, no tolerance change, and no SIGSEGV.
