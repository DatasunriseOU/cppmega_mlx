cmake_minimum_required(VERSION 3.27)

project(cppmega_native_optim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extension support library as shared" ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(cppmega_native_optim)
target_sources(
  cppmega_native_optim
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/fused_8bit.cpp)
target_include_directories(cppmega_native_optim PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cppmega_native_optim PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
      cppmega_native_optim_metallib
    TITLE
      cppmega_native_optim
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/fused_8bit.metal
    INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}
      ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(cppmega_native_optim cppmega_native_optim_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE cppmega_native_optim)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
