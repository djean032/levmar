cmake_minimum_required(VERSION 3.24)

project(levmar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(LEVMAR_BUILD_CONFORMANCE_RUNNER
       "Build the NIST conformance runner executable" ON)

add_library(levmar INTERFACE)
add_library(levmar::levmar ALIAS levmar)

target_compile_features(levmar INTERFACE cxx_std_23)
target_include_directories(
    levmar
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(LEVMAR_BUILD_CONFORMANCE_RUNNER)
  add_executable(levmar_nist_runner conformance/nist_nls/cpp_runner.cpp)
  target_link_libraries(levmar_nist_runner PRIVATE levmar::levmar)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(levmar_nist_runner PRIVATE -ffp-contract=off)
  endif()
endif()

install(TARGETS levmar EXPORT levmarTargets)

install(FILES levmar/lm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/levmar)

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

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/levmarConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/levmarConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/levmar)
