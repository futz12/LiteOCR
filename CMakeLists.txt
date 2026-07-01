cmake_minimum_required(VERSION 3.10)
project(LiteOCR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

include(GNUInstallDirs)

if(WIN32)
    add_compile_definitions(NOMINMAX)
    # stb_image_write uses sprintf; avoid MSVC deprecation warnings.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Top-level switch for ncnn Vulkan/GPU support.  Default OFF keeps wheels and
# CI free of Vulkan loader/runtime dependencies.
option(LITEOCR_ENABLE_VULKAN "Build LiteOCR with ncnn Vulkan support" OFF)

# Handle ncnn dependency (bundled submodule by default).
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(deps_ncnn)

# LiteOCR source files.
set(LITEOCR_SOURCES
    src/backend/liteocr_docinfer.cpp
    src/backend/liteocr_uvdoc.cpp
    src/backend/liteocr_paddleocrv5.cpp
    src/contours/liteocr_findcontours.cpp
    src/contours/liteocr_contourutils.cpp
    src/liteocr_image.cpp
    src/liteocr_imgproc.cpp
    src/liteocr_engine.cpp
    src/liteocr_c_api.cpp
    src/clipper.cpp
)

# LiteOCR static library.
add_library(LiteOCR STATIC ${LITEOCR_SOURCES})

target_include_directories(LiteOCR
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(LiteOCR PUBLIC ncnn)

# Optional shared library for Python ctypes wrapper.
option(LITEOCR_BUILD_SHARED "Build a shared LiteOCR library for the Python wrapper" OFF)
if(LITEOCR_BUILD_SHARED)
    add_library(LiteOCRShared SHARED ${LITEOCR_SOURCES})
    target_include_directories(LiteOCRShared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(LiteOCRShared PUBLIC ncnn)
    set_target_properties(LiteOCRShared PROPERTIES
        OUTPUT_NAME liteocr
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
    install(TARGETS LiteOCRShared
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Tests.
include(CTest)
foreach(test_name textlineori docori contours utils)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_include_directories(test_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_${test_name} PRIVATE LiteOCR ncnn)
    add_test(NAME test_${test_name} COMMAND test_${test_name})
endforeach()

set_tests_properties(test_textlineori test_docori PROPERTIES SKIP_RETURN_CODE 77)

# Examples.
foreach(example_name detector recognizer baseocr uvdoc slanet tableocr)
    add_executable(example_${example_name} examples/example_${example_name}.cpp)
    target_include_directories(example_${example_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(example_${example_name} PRIVATE LiteOCR ncnn)
endforeach()

add_executable(benchmark_ppocr examples/benchmark_ppocr.cpp)
target_include_directories(benchmark_ppocr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(benchmark_ppocr PRIVATE LiteOCR ncnn)

# Install rules.
if(NOT LITEOCR_BUILD_SHARED)
    install(TARGETS LiteOCR
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
install(FILES include/liteocr.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
