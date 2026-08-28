cmake_minimum_required(VERSION 3.24)

project(levmar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(LEVMAR_BUILD_CONFORMANCE_RUNNER
       "Build the NIST conformance runner executable" ON)
option(LEVMAR_BUILD_EXTERNAL_BENCHMARKS
       "Enable optional Ceres and CMinpack NIST solver benchmarks" OFF)
set(LEVMAR_BENCHMARK_COMPILE_OPTIONS "" CACHE STRING
    "Additional compile options for benchmark targets")
if(LEVMAR_BENCHMARK_COMPILE_OPTIONS)
  separate_arguments(levmar_benchmark_compile_options NATIVE_COMMAND
                     "${LEVMAR_BENCHMARK_COMPILE_OPTIONS}")
endif()

add_library(levmar INTERFACE)
add_library(levmar::levmar ALIAS levmar)

target_compile_features(levmar INTERFACE cxx_std_23)
target_include_directories(
    levmar
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(LEVMAR_BUILD_CONFORMANCE_RUNNER)
  function(levmar_add_nist_runner target)
    add_executable(${target} conformance/nist_nls/cpp_runner.cpp)
    target_link_libraries(${target} PRIVATE levmar::levmar)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${target} PRIVATE -ffp-contract=off)
    endif()
    if(LEVMAR_BENCHMARK_COMPILE_OPTIONS)
      target_compile_options(${target} PRIVATE
                             ${levmar_benchmark_compile_options})
    endif()
  endfunction()

  function(levmar_add_external_nist_runner target library definition)
    levmar_add_nist_runner(${target})
    target_link_libraries(${target} PRIVATE ${library})
    target_compile_definitions(${target} PRIVATE ${definition})

    add_library(${target}_pch OBJECT
                conformance/nist_nls/external_benchmark_support.cpp)
    target_link_libraries(${target}_pch PRIVATE levmar::levmar ${library})
    target_compile_definitions(${target}_pch PRIVATE ${definition})
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${target}_pch PRIVATE -ffp-contract=off)
    endif()
    if(LEVMAR_BENCHMARK_COMPILE_OPTIONS)
      target_compile_options(${target}_pch PRIVATE
                             ${levmar_benchmark_compile_options})
    endif()
    target_precompile_headers(${target}_pch PRIVATE ${ARGN})
    target_precompile_headers(${target} REUSE_FROM ${target}_pch)
  endfunction()

  levmar_add_nist_runner(levmar_nist_runner)
  set(levmar_nist_benchmark_runners levmar_nist_runner)

  if(LEVMAR_BUILD_EXTERNAL_BENCHMARKS)
    find_package(Ceres CONFIG QUIET)
    find_package(CMinpack CONFIG QUIET)
    if(Ceres_FOUND)
      levmar_add_external_nist_runner(
        levmar_nist_ceres_runner
        Ceres::ceres
        LEVMAR_HAVE_CERES=1
        <ceres/autodiff_cost_function.h>
        <ceres/ceres.h>)
      list(APPEND levmar_nist_benchmark_runners levmar_nist_ceres_runner)
    else()
      message(STATUS "Ceres not found; Ceres benchmarks are disabled")
    endif()
    if(CMinpack_FOUND)
      levmar_add_external_nist_runner(
        levmar_nist_minpack_runner
        cminpack::cminpack
        LEVMAR_HAVE_CMINPACK=1
        <cminpack.h>)
      list(APPEND levmar_nist_benchmark_runners levmar_nist_minpack_runner)
    else()
      message(STATUS "CMinpack not found; MINPACK benchmarks are disabled")
    endif()
  endif()

  add_custom_target(levmar_nist_benchmark_runners
                    DEPENDS ${levmar_nist_benchmark_runners})
endif()

add_executable(levmar_autodiff_graph_tests tests/autodiff_graph_tests.cpp)
target_link_libraries(levmar_autodiff_graph_tests PRIVATE levmar::levmar)

add_test(NAME levmar_autodiff_graph_tests COMMAND levmar_autodiff_graph_tests)

install(TARGETS levmar EXPORT levmarTargets)

install(
    DIRECTORY levmar/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/levmar
    FILES_MATCHING
    PATTERN "*.h")

install(
    EXPORT levmarTargets
    FILE levmarTargets.cmake
    NAMESPACE levmar::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/levmar)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/levmarConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/levmarConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/levmarConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/levmar)

install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/levmarConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/levmarConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/levmar)
