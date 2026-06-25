# PR / discussion focus — the @sha/@repo/@pr/@details docstring at the head of each block


---

## PR/commit head — `s2n-tls_r2500.parquet` row 0
### Extracted PR/discussion fields
```json
{
  "brief": "Support wildcard matching to select server certs",
  "pr_number": null,
  "sha_in_doc": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "crypto/s2n_certificate.c",
  "commit_hash": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "timestamp": "2019-05-28T13:51:00-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
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
### Block head (docstring -> first code lines)
```cpp
a simple wildcard
 * certificate that matches a domain name, we will prefer the exact match.
 * 
 * TODO:
 * - [] Add more test cases for wildcardify-ing
 * - [] Update fuzz test to cover wildcard
 */
// === PRE-COMMIT ===
int s2n_cert_public_key_set_rsa_from_openssl(s2n_cert_public_key *public_key, RSA *openssl_rsa)
{
 notnull_check(openssl_rsa);
 notnull_c
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 1
### Extracted PR/discussion fields
```json
{
  "brief": "Support wildcard matching to select server certs",
  "pr_number": null,
  "sha_in_doc": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "crypto/s2n_certificate.c",
  "commit_hash": "032d8f3d707c75e98a1a211315f1173ec93fa918",
  "timestamp": "2019-05-28T13:51:00-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
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
### Block head (docstring -> first code lines)
```cpp
a simple wildcard
 * certificate that matches a domain name, we will prefer the exact match.
 * 
 * TODO:
 * - [] Add more test cases for wildcardify-ing
 * - [] Update fuzz test to cover wildcard
 */
// === CONTEXT ===
int s2n_cert_public_key_set_rsa_from_openssl(s2n_cert_public_key *public_key, RSA *openssl_rsa)
{
 notnull_check(openssl_rsa);
 notnull_chec
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 2
### Extracted PR/discussion fields
```json
{
  "brief": "Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length",
  "pr_number": null,
  "sha_in_doc": "3e91e4531bad3bf9173c04a768856504931382b5",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/cleanup.h",
  "commit_hash": "3e91e4531bad3bf9173c04a768856504931382b5",
  "timestamp": "2019-05-20T11:52:46-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
```
### Block head (docstring -> first code lines)
```cpp

/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
// === PRE-COMMIT ===
_INLINE_ void compressed_idx_dv_ar_cleanup(IN OUT compressed_idx_dv_ar_t *o) 
{
 for(int i=0; i < N0; i++)
 {
 secure_clean((uint8_t*)&o[
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 3
### Extracted PR/discussion fields
```json
{
  "brief": "Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length",
  "pr_number": null,
  "sha_in_doc": "3e91e4531bad3bf9173c04a768856504931382b5",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/cleanup.h",
  "commit_hash": "3e91e4531bad3bf9173c04a768856504931382b5",
  "timestamp": "2019-05-20T11:52:46-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
```
### Block head (docstring -> first code lines)
```cpp

/**
 * @brief Fix BIKE compressed_idx_dv_ar_cleanup to cleanup the correct memory length
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/cleanup.h
 * @sha 3e91e4531bad3bf9173c04a768856504931382b5
 */
// === CONTEXT ===
_INLINE_ void compressed_idx_dv_ar_cleanup(IN OUT compressed_idx_dv_ar_t *o) 
{
 for(int i=0; i < N0; i++)
 {
 secure_clean((uint8_t*)&o[i],
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 4
### Extracted PR/discussion fields
```json
{
  "brief": "Use inttypes macro to print uint64_t",
  "pr_number": null,
  "sha_in_doc": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/utilities.c",
  "commit_hash": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "timestamp": "2019-05-17T21:13:33-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
```
### Block head (docstring -> first code lines)
```cpp
 c++17
// arch: x86_64
// mode: user
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
// === PRE-COMMIT ===
_INLINE_ void print_uint64(IN const uint64_t val)
{
 // If printing in BE is requried swap the order of bytes
#ifdef PRINT_IN_BE
 uint64_
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 5
### Extracted PR/discussion fields
```json
{
  "brief": "Use inttypes macro to print uint64_t",
  "pr_number": null,
  "sha_in_doc": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "pq-crypto/bike/utilities.c",
  "commit_hash": "5cd6e4edff395b12f6e020ccb605597fa50548b6",
  "timestamp": "2019-05-17T21:13:33-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
```
### Block head (docstring -> first code lines)
```cpp
 c++17
// arch: x86_64
// mode: user
/**
 * @brief Use inttypes macro to print uint64_t
 *
 * @repo aws/s2n-tls
 * File: pq-crypto/bike/utilities.c
 * @sha 5cd6e4edff395b12f6e020ccb605597fa50548b6
 */
// === CONTEXT ===
_INLINE_ void print_uint64(IN const uint64_t val)
{
 // If printing in BE is requried swap the order of bytes
#ifdef PRINT_IN_BE
 uint64_t t
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 6
### Extracted PR/discussion fields
```json
{
  "brief": "Store certificate san_names in s2n_array",
  "pr_number": null,
  "sha_in_doc": "ae9c2efb95474251d1f2815f433af6b3c66d33ae",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "crypto/s2n_certificate.h",
  "commit_hash": "ae9c2efb95474251d1f2815f433af6b3c66d33ae",
  "timestamp": "2019-05-17T11:05:40-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Store certificate san_names in s2n_array
 *
 * @repo aws/s2n-tls
 * File: crypto/s2n_certificate.h
 * @sha ae9c2efb95474251d1f2815f433af6b3c66d33ae
 *
 * @details
 * And avoid holding onto a GENERAL_NAMES and X509 object for the
 * lifetime of the s2n_cert_chain_and_key.
 */
```
### Block head (docstring -> first code lines)
```cpp
rypto/s2n_certificate.h
 * @sha ae9c2efb95474251d1f2815f433af6b3c66d33ae
 *
 * @details
 * And avoid holding onto a GENERAL_NAMES and X509 object for the
 * lifetime of the s2n_cert_chain_and_key.
 */
// === PRE-COMMIT ===
typedef enum {
 S2N_AUTHENTICATION_RSA = 0,
 S2N_AUTHENTICATION_ECDSA,
 S2N_AUTHENTICATION_METHOD_SENTINEL
} s2n_authentication_method
st
```

---

## PR/commit head — `s2n-tls_r2500.parquet` row 7
### Extracted PR/discussion fields
```json
{
  "brief": "Support CN for server cert selection",
  "pr_number": null,
  "sha_in_doc": "93abe1d96ad99f28d72996c8a91fae82ffde4f6a",
  "repo_in_doc": "aws/s2n-tls",
  "repo": "aws/s2n-tls",
  "filepath": "crypto/s2n_certificate.h",
  "commit_hash": "93abe1d96ad99f28d72996c8a91fae82ffde4f6a",
  "timestamp": "2019-05-17T11:05:40-07:00"
}
```
### Raw docstring (sits at the HEAD of the atomic block, before PRE code)
```c
/**
 * @brief Support CN for server cert selection
 *
 * @repo aws/s2n-tls
 * File: crypto/s2n_certificate.h
 * @sha 93abe1d96ad99f28d72996c8a91fae82ffde4f6a
 *
 * @details
 * After this change, s2n will use the CommonName entry from the Subject of
 * to select from certificates added to s2n_config. Usage of CN has been
 * deprecated since RFC2818 in favor of SAN. This change only uses the CN
 * if no valid SANs are available in the cert.
 * 
 * Specifics:
 * - Multiple CommonNames are supported, though practically usage should be
 * very rare. A CAB thread on dropping support for it:
 * https://cabforum.org/pipermail/public/2016-April/007242.html .
 */
```
### Block head (docstring -> first code lines)
```cpp
 * - Multiple CommonNames are supported, though practically usage should be
 * very rare. A CAB thread on dropping support for it:
 * https://cabforum.org/pipermail/public/2016-April/007242.html .
 */
// === PRE-COMMIT ===
typedef enum {
 S2N_AUTHENTICATION_RSA = 0,
 S2N_AUTHENTICATION_ECDSA,
 S2N_AUTHENTICATION_METHOD_SENTINEL
} s2n_authentication_method
st
```