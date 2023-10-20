add_custom_target(trs4slip_python)

function(find_python_module NAME)
  message("Checking Python module ${NAME}")
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import ${NAME}"
    RESULT_VARIABLE "_MODULE_FOUND_RETURN_CODE"
    OUTPUT_QUIET
    ERROR_QUIET)

  if(${_MODULE_FOUND_RETURN_CODE} EQUAL "0")
    set("PYTHON_MODULE_${NAME}_FOUND" 1)
    message("Found Python module ${NAME}")
  else()
    set("PYTHON_MODULE_${NAME}_FOUND" 0)
    message("Could NOT find Python module ${NAME}")
  endif()
endfunction()

find_python_module(build)

if(NOT ${PYTHON_MODULE_build_FOUND})
  message(FATAL_ERROR "Could NOT find required Python module build")
endif()

add_custom_command(TARGET trs4slip_python PRE_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
  "${PROJECT_SOURCE_DIR}/src/trs4slip.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/trs4slip.h")

add_custom_command(TARGET trs4slip_python PRE_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
  "${PROJECT_SOURCE_DIR}/src/trs4slip.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/trs4slip.cpp")

add_custom_target(trs4slip_python_sdist)
add_custom_target(trs4slip_python_whl)

add_dependencies(trs4slip_python_sdist trs4slip_python)
add_dependencies(trs4slip_python_whl trs4slip_python)

add_custom_command(TARGET trs4slip_python_sdist POST_BUILD
  COMMAND Python::Interpreter -m build -s
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(TARGET trs4slip_python_whl POST_BUILD
  COMMAND Python::Interpreter -m build -w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
