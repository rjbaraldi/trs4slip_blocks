add_library(
	trs4slip SHARED
	trs4slip.cpp
)

target_include_directories(
	trs4slip PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(trs4slip)

set_target_properties(
	trs4slip
	PROPERTIES PUBLIC_HEADER "trs4slip.h"
)

install(
	TARGETS trs4slip
  EXPORT trs4slip-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT trs4slip-targets
  FILE "${PROJECT_NAME}-targets.cmake"
  NAMESPACE
  "${PROJECT_NAME}::"
  DESTINATION
  "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

add_executable(
	trs4slip_test
	trs4slip_test.cpp
)

target_link_libraries(
	trs4slip_test 
	trs4slip
)

add_test(
	NAME trs4slip_test
	COMMAND trs4slip_test
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
