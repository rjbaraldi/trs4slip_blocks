cmake_minimum_required(VERSION 3.12)
project(trs4slip
  VERSION 0.1.0
  LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

add_subdirectory(src)

find_package(Python "3.10"
  COMPONENTS Interpreter)

if(${Python_FOUND})
  add_subdirectory(python)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/trs4slip.pc.in"
  "${PROJECT_BINARY_DIR}/trs4slip.pc"
  @ONLY)

install(
  FILES
  "${PROJECT_BINARY_DIR}/trs4slip.pc"
  DESTINATION
  "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
