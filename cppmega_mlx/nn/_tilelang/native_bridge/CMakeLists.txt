cmake_minimum_required(VERSION 3.26)
project(cppmega_tilelang_mlx_tvm_ffi_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TILELANG_ROOT "" CACHE PATH "TileLang source or install root")
set(TILELANG_MLX_TVM_FFI_LIBRARY "" CACHE FILEPATH
    "Linkable TileLang MLX TVM-FFI C API library")
set(CPPMEGA_MLX_PYTHON_BRIDGE "" CACHE FILEPATH
    "Optional libmlx_python_bridge.dylib override for status checks")
set(CPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_VERSION "1" CACHE STRING
    "TileLang MLX TVM-FFI C API version cppmega was audited against")
set(CPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_ABI_HASH
    "tilelang.mlx_tvm_ffi.c_api.v1.pyobject_metal_call.20260514" CACHE STRING
    "TileLang MLX TVM-FFI C API ABI hash cppmega was audited against")

function(cppmega_set_bridge_status status todo)
  set(CPPMEGA_TILELANG_MLX_TVM_FFI_STATUS "${status}" CACHE STRING
      "cppmega native TileLang MLX TVM-FFI bridge status" FORCE)
  set(CPPMEGA_TILELANG_MLX_TVM_FFI_TODO "${todo}" CACHE STRING
      "next action for cppmega native TileLang MLX TVM-FFI bridge" FORCE)
  message(STATUS "cppmega TileLang MLX TVM-FFI bridge status: ${status}")
  message(STATUS "cppmega TileLang MLX TVM-FFI bridge TODO: ${todo}")
endfunction()

function(cppmega_nm_contains output_var path symbol)
  find_program(CPPMEGA_NM_EXECUTABLE nm)
  if(NOT CPPMEGA_NM_EXECUTABLE)
    set(${output_var} FALSE PARENT_SCOPE)
    return()
  endif()
  if(APPLE)
    execute_process(
      COMMAND "${CPPMEGA_NM_EXECUTABLE}" -gU "${path}"
      OUTPUT_VARIABLE _cppmega_nm_out
      ERROR_VARIABLE _cppmega_nm_err
      RESULT_VARIABLE _cppmega_nm_rc)
  else()
    execute_process(
      COMMAND "${CPPMEGA_NM_EXECUTABLE}" -D "${path}"
      OUTPUT_VARIABLE _cppmega_nm_out
      ERROR_VARIABLE _cppmega_nm_err
      RESULT_VARIABLE _cppmega_nm_rc)
  endif()
  if(_cppmega_nm_rc EQUAL 0 AND _cppmega_nm_out MATCHES "${symbol}")
    set(${output_var} TRUE PARENT_SCOPE)
  else()
    set(${output_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(cppmega_binary_contains output_var path needle)
  find_program(CPPMEGA_STRINGS_EXECUTABLE strings)
  if(NOT CPPMEGA_STRINGS_EXECUTABLE)
    set(${output_var} FALSE PARENT_SCOPE)
    return()
  endif()
  execute_process(
    COMMAND "${CPPMEGA_STRINGS_EXECUTABLE}" "${path}"
    OUTPUT_VARIABLE _cppmega_strings_out
    ERROR_VARIABLE _cppmega_strings_err
    RESULT_VARIABLE _cppmega_strings_rc)
  if(_cppmega_strings_rc EQUAL 0)
    string(FIND "${_cppmega_strings_out}" "${needle}" _cppmega_needle_pos)
    if(NOT _cppmega_needle_pos EQUAL -1)
      set(${output_var} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${output_var} FALSE PARENT_SCOPE)
endfunction()

if(APPLE)
  if(CPPMEGA_MLX_PYTHON_BRIDGE)
    set(_cppmega_mlx_python_bridge "${CPPMEGA_MLX_PYTHON_BRIDGE}")
  else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import pathlib, sys; import mlx; root=pathlib.Path(next(iter(mlx.__path__))); print(root / 'lib' / 'libmlx_python_bridge.dylib')"
        OUTPUT_VARIABLE _cppmega_mlx_python_bridge
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _cppmega_mlx_probe_rc)
    endif()
  endif()

  if(NOT _cppmega_mlx_python_bridge OR NOT EXISTS "${_cppmega_mlx_python_bridge}")
    cppmega_set_bridge_status(
      "blocked_missing_mlx_python_bridge"
      "Install or build the MLX fork that ships libmlx_python_bridge.dylib exporting mlx_core_wrap_mx_array_move. Do not add a production dlsym or mx.fast.metal_kernel fallback.")
    return()
  endif()
  file(SHA256 "${_cppmega_mlx_python_bridge}" _cppmega_mlx_python_bridge_sha256)

  cppmega_nm_contains(_cppmega_has_mlx_wrap_symbol
    "${_cppmega_mlx_python_bridge}"
    "mlx_core_wrap_mx_array_move")
  if(NOT _cppmega_has_mlx_wrap_symbol)
    cppmega_set_bridge_status(
      "blocked_missing_mlx_wrap_symbol"
      "The active MLX Python bridge does not export mlx_core_wrap_mx_array_move. Rebuild the MLX fork with the native array wrapper symbol before enabling cppmega direct linking.")
    return()
  endif()
endif()

set(_cppmega_tilelang_include_hints)
if(TILELANG_ROOT)
  list(APPEND _cppmega_tilelang_include_hints
       "${TILELANG_ROOT}/tilelang/include"
       "${TILELANG_ROOT}/src")
endif()
find_path(TILELANG_MLX_TVM_FFI_INCLUDE_DIR
  NAMES
    tilelang/contrib/mlx_tvm_ffi_c_api.h
    contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h
  HINTS ${_cppmega_tilelang_include_hints})
if(NOT TILELANG_MLX_TVM_FFI_INCLUDE_DIR)
  cppmega_set_bridge_status(
    "blocked_missing_tilelang_c_api_header"
    "Rebuild or install TileLang with src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h from the native C API patch.")
  return()
endif()
foreach(_cppmega_tilelang_c_api_header_rel
        "tilelang/contrib/mlx_tvm_ffi_c_api.h"
        "contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h")
  if(EXISTS "${TILELANG_MLX_TVM_FFI_INCLUDE_DIR}/${_cppmega_tilelang_c_api_header_rel}")
    set(_cppmega_tilelang_c_api_header
        "${TILELANG_MLX_TVM_FFI_INCLUDE_DIR}/${_cppmega_tilelang_c_api_header_rel}")
    break()
  endif()
endforeach()
if(NOT _cppmega_tilelang_c_api_header)
  cppmega_set_bridge_status(
    "blocked_missing_tilelang_c_api_header"
    "CMake found a TileLang include root but not mlx_tvm_ffi_c_api.h under that root.")
  return()
endif()
file(READ "${_cppmega_tilelang_c_api_header}" _cppmega_tilelang_c_api_header_text)
string(REPLACE "\n" " " _cppmega_tilelang_c_api_header_flat
       "${_cppmega_tilelang_c_api_header_text}")
string(REGEX MATCH
       "#define[ \t]+TILELANG_MLX_TVM_FFI_C_API_VERSION[ \t]+([0-9]+)u?"
       _cppmega_tilelang_c_api_version_match
       "${_cppmega_tilelang_c_api_header_flat}")
if(NOT _cppmega_tilelang_c_api_version_match)
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_version_unknown"
    "TileLang MLX TVM-FFI C API header does not expose TILELANG_MLX_TVM_FFI_C_API_VERSION.")
  return()
endif()
set(_cppmega_tilelang_c_api_version "${CMAKE_MATCH_1}")
if(NOT "${_cppmega_tilelang_c_api_version}" STREQUAL
       "${CPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_VERSION}")
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_version_mismatch"
    "TileLang MLX TVM-FFI C API version changed. Audit cppmega native bridge before linking.")
  return()
endif()
string(REGEX MATCH
       "TILELANG_MLX_TVM_FFI_C_API_ABI_HASH[^\"]*\"([^\"]+)\""
       _cppmega_tilelang_c_api_abi_match
       "${_cppmega_tilelang_c_api_header_flat}")
if(NOT _cppmega_tilelang_c_api_abi_match)
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_abi_unknown"
    "TileLang MLX TVM-FFI C API header does not expose TILELANG_MLX_TVM_FFI_C_API_ABI_HASH.")
  return()
endif()
set(_cppmega_tilelang_c_api_abi_hash "${CMAKE_MATCH_1}")
if(NOT "${_cppmega_tilelang_c_api_abi_hash}" STREQUAL
       "${CPPMEGA_EXPECTED_TILELANG_MLX_TVM_FFI_C_API_ABI_HASH}")
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_abi_mismatch"
    "TileLang MLX TVM-FFI C API ABI hash changed. Audit cppmega native bridge before linking.")
  return()
endif()
file(SHA256 "${_cppmega_tilelang_c_api_header}"
     _cppmega_tilelang_c_api_header_sha256)

if(NOT TILELANG_MLX_TVM_FFI_LIBRARY AND TILELANG_ROOT)
  file(GLOB _cppmega_tilelang_c_api_candidates
       LIST_DIRECTORIES FALSE
       "${TILELANG_ROOT}/build/lib/libtilelang_mlx_tvm_ffi_c_api.dylib"
       "${TILELANG_ROOT}/build/lib/libtilelang_mlx_tvm_ffi_c_api.so"
       "${TILELANG_ROOT}/lib/libtilelang_mlx_tvm_ffi_c_api.dylib"
       "${TILELANG_ROOT}/lib/libtilelang_mlx_tvm_ffi_c_api.so"
       "${TILELANG_ROOT}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.dylib"
       "${TILELANG_ROOT}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.so")
  list(SORT _cppmega_tilelang_c_api_candidates)
  list(LENGTH _cppmega_tilelang_c_api_candidates _cppmega_candidate_count)
  if(_cppmega_candidate_count GREATER 0)
    list(GET _cppmega_tilelang_c_api_candidates 0 TILELANG_MLX_TVM_FFI_LIBRARY)
  endif()
endif()

if(NOT TILELANG_MLX_TVM_FFI_LIBRARY OR NOT EXISTS "${TILELANG_MLX_TVM_FFI_LIBRARY}")
  cppmega_set_bridge_status(
    "blocked_missing_tilelang_c_api_library"
    "Build TileLang after applying the native C API patch so libtilelang_mlx_tvm_ffi_c_api is available for direct linking.")
  return()
endif()

cppmega_nm_contains(_cppmega_has_tilelang_c_api_symbol
  "${TILELANG_MLX_TVM_FFI_LIBRARY}"
  "tilelang_mlx_tvm_ffi_get_c_api")
if(NOT _cppmega_has_tilelang_c_api_symbol)
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_symbol_missing"
    "The TileLang library exists but does not export tilelang_mlx_tvm_ffi_get_c_api. Rebuild TileLang with the native C API patch.")
  return()
endif()
cppmega_binary_contains(_cppmega_library_has_abi_hash
  "${TILELANG_MLX_TVM_FFI_LIBRARY}"
  "${_cppmega_tilelang_c_api_abi_hash}")
if(NOT _cppmega_library_has_abi_hash)
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_library_abi_mismatch"
    "TileLang C API library does not contain the ABI hash from the selected header. Rebuild TileLang and avoid mixing headers/libraries.")
  return()
endif()
cppmega_binary_contains(_cppmega_library_has_header_sha
  "${TILELANG_MLX_TVM_FFI_LIBRARY}"
  "${_cppmega_tilelang_c_api_header_sha256}")
if(NOT _cppmega_library_has_header_sha)
  cppmega_set_bridge_status(
    "blocked_tilelang_c_api_library_header_mismatch"
    "TileLang C API library was built with a different mlx_tvm_ffi_c_api.h than the selected include path.")
  return()
endif()
if(APPLE)
  cppmega_binary_contains(_cppmega_library_has_mlx_bridge_sha
    "${TILELANG_MLX_TVM_FFI_LIBRARY}"
    "${_cppmega_mlx_python_bridge_sha256}")
  if(NOT _cppmega_library_has_mlx_bridge_sha)
    cppmega_set_bridge_status(
      "blocked_tilelang_c_api_mlx_bridge_mismatch"
      "TileLang C API library was built against a different libmlx_python_bridge.dylib than the active MLX package.")
    return()
  endif()
endif()

add_library(cppmega_tilelang_mlx_tvm_ffi_bridge SHARED
  tilelang_mlx_tvm_ffi_bridge.cpp)
target_compile_definitions(cppmega_tilelang_mlx_tvm_ffi_bridge PRIVATE
  CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORTS)
target_include_directories(cppmega_tilelang_mlx_tvm_ffi_bridge PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}"
  "${TILELANG_MLX_TVM_FFI_INCLUDE_DIR}")
target_link_libraries(cppmega_tilelang_mlx_tvm_ffi_bridge PRIVATE
  "${TILELANG_MLX_TVM_FFI_LIBRARY}")

get_filename_component(_cppmega_tilelang_c_api_dir
  "${TILELANG_MLX_TVM_FFI_LIBRARY}" DIRECTORY)
set_target_properties(cppmega_tilelang_mlx_tvm_ffi_bridge PROPERTIES
  BUILD_RPATH "${_cppmega_tilelang_c_api_dir}"
  INSTALL_RPATH "@loader_path;${_cppmega_tilelang_c_api_dir}")

cppmega_set_bridge_status(
  "available"
  "cppmega_tilelang_mlx_tvm_ffi_bridge links directly to TileLang's native C API; production Python callsites route through the standard TileLang tvm-ffi adapter.")
