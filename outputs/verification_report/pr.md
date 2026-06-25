# PR / discussion focus — the @sha/@repo/@pr/@details docstring at the head of each block


---

## PR/commit head — `s2n-tls_r0.parquet` row 0
### Extracted PR/discussion fields
```json
{
  "brief": "fix: make get_alert idempotent (#5767)",
  "pr_number": "5767",
  "sha_in_doc": "5f54315df13a7399136c5093ab43fa24f5411632",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "tls/s2n_connection.c",
  "commit_hash": "5f54315df13a7399136c5093ab43fa24f5411632",
  "timestamp": "2026-03-04T21:43:13Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief fix: make get_alert idempotent (#5767) * * @repo _src * File: tls/s2n_connection.c * @sha 5f54315df13a7399136c5093ab43fa24f5411632 * @pr 5767 */
```
### Block head (docstring -> first code lines)
```cpp
standard: c++17// arch: x86_64// mode: user/** * @brief fix: make get_alert idempotent (#5767) * * @repo _src * File: tls/s2n_connection.c * @sha 5f54315df13a7399136c5093ab43fa24f5411632 * @pr 5767 */// === PRE-COMMIT ===int s2n_connection_get_alert(struct s2n_connection *conn){    POSIX_ENSURE_REF(conn);    S2N_ERROR_IF(s2n_stuffer_data_available(&conn->ale
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 1
### Extracted PR/discussion fields
```json
{
  "brief": "fix: incorrect group reported for TLS 1.2 session resumption (#5673)",
  "pr_number": "5673",
  "sha_in_doc": "6b5b32abde98f1a00c414e1a3217465f6a37fa44",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "tls/s2n_connection.c",
  "commit_hash": "6b5b32abde98f1a00c414e1a3217465f6a37fa44",
  "timestamp": "2025-12-31T00:54:41Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief fix: incorrect group reported for TLS 1.2 session resumption (#5673) * * @repo _src * File: tls/s2n_connection.c * @sha 6b5b32abde98f1a00c414e1a3217465f6a37fa44 * @pr 5673 * * @details */
```
### Block head (docstring -> first code lines)
```cpp
/** * @brief fix: incorrect group reported for TLS 1.2 session resumption (#5673) * * @repo _src * File: tls/s2n_connection.c * @sha 6b5b32abde98f1a00c414e1a3217465f6a37fa44 * @pr 5673 * * @details */// === PRE-COMMIT ===const char *s2n_connection_get_curve(struct s2n_connection *conn){    PTR_ENSURE_REF(conn);    PTR_ENSURE_REF(conn->secure);    PTR_ENSURE_
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 2
### Extracted PR/discussion fields
```json
{
  "brief": "ci: update clang format version (#5661)",
  "pr_number": "5661",
  "sha_in_doc": "a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "crypto/s2n_openssl_evp.h",
  "commit_hash": "a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402",
  "timestamp": "2025-12-11T18:44:07Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief ci: update clang format version (#5661) * * @repo _src * File: crypto/s2n_openssl_evp.h * @sha a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402 * @pr 5661 */
```
### Block head (docstring -> first code lines)
```cpp
ard: c++17// arch: x86_64// mode: user/** * @brief ci: update clang format version (#5661) * * @repo _src * File: crypto/s2n_openssl_evp.h * @sha a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402 * @pr 5661 */// === PRE-COMMIT ===DEFINE_POINTER_CLEANUP_FUNC(EVP_PKEY*, EVP_PKEY_free)// === POST-COMMIT: ci: update clang format version (#5661) ===DEFINE_POINTER_CLEANUP_
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 3
### Extracted PR/discussion fields
```json
{
  "brief": "ci: update clang format version (#5661)",
  "pr_number": "5661",
  "sha_in_doc": "a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "crypto/s2n_openssl_evp.h",
  "commit_hash": "a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402",
  "timestamp": "2025-12-11T18:44:07Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief ci: update clang format version (#5661) * * @repo _src * File: crypto/s2n_openssl_evp.h * @sha a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402 * @pr 5661 */
```
### Block head (docstring -> first code lines)
```cpp
ard: c++17// arch: x86_64// mode: user/** * @brief ci: update clang format version (#5661) * * @repo _src * File: crypto/s2n_openssl_evp.h * @sha a7bdb88ebad5e29c53ac098a1ae6e6c5cd737402 * @pr 5661 */// === CONTEXT ===DEFINE_POINTER_CLEANUP_FUNC(EVP_PKEY*, EVP_PKEY_free)// === DIFF ===diff --git a/crypto/s2n_openssl_evp.h b/crypto/s2n_openssl_evp.hindex 4b03
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 4
### Extracted PR/discussion fields
```json
{
  "brief": "feat: improve performance of getting validated cert chain from libcrypto (#5622)",
  "pr_number": "5622",
  "sha_in_doc": "0ffb435c02e95877da27aaa50898bfe0b9eaf057",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "tls/s2n_x509_validator.h",
  "commit_hash": "0ffb435c02e95877da27aaa50898bfe0b9eaf057",
  "timestamp": "2025-11-29T00:28:31Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief feat: improve performance of getting validated cert chain from libcrypto (#5622) * * @repo _src * File: tls/s2n_x509_validator.h * @sha 0ffb435c02e95877da27aaa50898bfe0b9eaf057 * @pr 5622 */
```
### Block head (docstring -> first code lines)
```cpp
 * @brief feat: improve performance of getting validated cert chain from libcrypto (#5622) * * @repo _src * File: tls/s2n_x509_validator.h * @sha 0ffb435c02e95877da27aaa50898bfe0b9eaf057 * @pr 5622 */// === PRE-COMMIT ===typedef enum {    UNINIT,    INIT,    READY_TO_VERIFY,    AWAITING_CRL_CALLBACK,    VALIDATED,    OCSP_VALIDATED,} validator_statetypedef u
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 5
### Extracted PR/discussion fields
```json
{
  "brief": "feat: improve performance of getting validated cert chain from libcrypto (#5622)",
  "pr_number": "5622",
  "sha_in_doc": "0ffb435c02e95877da27aaa50898bfe0b9eaf057",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "tls/s2n_x509_validator.h",
  "commit_hash": "0ffb435c02e95877da27aaa50898bfe0b9eaf057",
  "timestamp": "2025-11-29T00:28:31Z"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief feat: improve performance of getting validated cert chain from libcrypto (#5622) * * @repo _src * File: tls/s2n_x509_validator.h * @sha 0ffb435c02e95877da27aaa50898bfe0b9eaf057 * @pr 5622 */
```
### Block head (docstring -> first code lines)
```cpp
 * @brief feat: improve performance of getting validated cert chain from libcrypto (#5622) * * @repo _src * File: tls/s2n_x509_validator.h * @sha 0ffb435c02e95877da27aaa50898bfe0b9eaf057 * @pr 5622 */// === CONTEXT ===typedef enum {    UNINIT,    INIT,    READY_TO_VERIFY,    AWAITING_CRL_CALLBACK,    VALIDATED,    OCSP_VALIDATED,} validator_statetypedef uint
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 6
### Extracted PR/discussion fields
```json
{
  "brief": "feat: add ML-KEM-1024 kem definition (#5367)",
  "pr_number": "5367",
  "sha_in_doc": "4bb511884a886ada7ef422c6e1741252a582f128",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "crypto/s2n_fips_rules.c",
  "commit_hash": "4bb511884a886ada7ef422c6e1741252a582f128",
  "timestamp": "2025-07-11T11:03:38-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief feat: add ML-KEM-1024 kem definition (#5367) * * @repo _src * File: crypto/s2n_fips_rules.c * @sha 4bb511884a886ada7ef422c6e1741252a582f128 * @pr 5367 */
```
### Block head (docstring -> first code lines)
```cpp
 c++17// arch: x86_64// mode: user/** * @brief feat: add ML-KEM-1024 kem definition (#5367) * * @repo _src * File: crypto/s2n_fips_rules.c * @sha 4bb511884a886ada7ef422c6e1741252a582f128 * @pr 5367 */// === PRE-COMMIT ===S2N_RESULT s2n_fips_validate_kem(const struct s2n_kem *kem, bool *valid){    RESULT_ENSURE_REF(kem);    RESULT_ENSURE_REF(valid);    *valid
```

---

## PR/commit head — `s2n-tls_r0.parquet` row 7
### Extracted PR/discussion fields
```json
{
  "brief": "feat: add ML-KEM-1024 kem definition (#5367)",
  "pr_number": "5367",
  "sha_in_doc": "4bb511884a886ada7ef422c6e1741252a582f128",
  "repo_in_doc": "_src",
  "repo": "_src",
  "filepath": "crypto/s2n_fips_rules.c",
  "commit_hash": "4bb511884a886ada7ef422c6e1741252a582f128",
  "timestamp": "2025-07-11T11:03:38-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/** * @brief feat: add ML-KEM-1024 kem definition (#5367) * * @repo _src * File: crypto/s2n_fips_rules.c * @sha 4bb511884a886ada7ef422c6e1741252a582f128 * @pr 5367 */
```
### Block head (docstring -> first code lines)
```cpp
 c++17// arch: x86_64// mode: user/** * @brief feat: add ML-KEM-1024 kem definition (#5367) * * @repo _src * File: crypto/s2n_fips_rules.c * @sha 4bb511884a886ada7ef422c6e1741252a582f128 * @pr 5367 */// === CONTEXT ===S2N_RESULT s2n_fips_validate_kem(const struct s2n_kem *kem, bool *valid){    RESULT_ENSURE_REF(kem);    RESULT_ENSURE_REF(valid);    *valid = 
```