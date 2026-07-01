# deps_ncnn.cmake
# Handle ncnn dependency: either use a system-installed ncnn or build the
# bundled ncnn submodule. This file is inspired by zimage-ncnn-vulkan.

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)

if(USE_SYSTEM_NCNN)
    # ncnn with Vulkan needs glslang/SPIRV targets to be available.
    find_package(Threads)
    find_package(SPIRV-Tools QUIET)
    find_package(SPIRV-Tools-opt QUIET)
    find_package(glslang QUIET)

    if(glslang_FOUND)
        if(NOT TARGET glslang AND TARGET glslang::glslang)
            add_library(glslang ALIAS glslang::glslang)
        endif()
        if(NOT TARGET SPIRV AND TARGET glslang::SPIRV)
            add_library(SPIRV ALIAS glslang::SPIRV)
        endif()
    else()
        set(GLSLANG_TARGET_DIR "GLSLANG-NOTFOUND" CACHE PATH "Absolute path to glslangTargets.cmake directory")
        if(NOT GLSLANG_TARGET_DIR AND NOT DEFINED ENV{GLSLANG_TARGET_DIR})
            message(WARNING "set glslang_DIR to glslang-config.cmake directory for using system glslang.")
            message(WARNING "GLSLANG_TARGET_DIR must be defined! USE_SYSTEM_NCNN will be turned off.")
            set(USE_SYSTEM_NCNN OFF)
        else()
            include("${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake")
            include("${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake")
            if(EXISTS "${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
                include("${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
            endif()
            include("${GLSLANG_TARGET_DIR}/glslangTargets.cmake")
            include("${GLSLANG_TARGET_DIR}/SPIRVTargets.cmake")
        endif()
    endif()

    if(TARGET glslang AND TARGET SPIRV)
        get_property(glslang_location TARGET glslang PROPERTY LOCATION)
        get_property(SPIRV_location TARGET SPIRV PROPERTY LOCATION)
        message(STATUS "Found glslang: ${glslang_location}")
        message(STATUS "Found SPIRV: ${SPIRV_location}")
    else()
        if(LITEOCR_ENABLE_VULKAN)
            message(WARNING "LITEOCR_ENABLE_VULKAN=ON requires glslang/SPIRV, but they were not found. USE_SYSTEM_NCNN will be turned off.")
        else()
            message(WARNING "glslang or SPIRV target not found! USE_SYSTEM_NCNN will be turned off.")
        endif()
        set(USE_SYSTEM_NCNN OFF)
    endif()
endif()

if(USE_SYSTEM_NCNN)
    find_package(ncnn)
    if(NOT TARGET ncnn)
        message(WARNING "ncnn target not found! USE_SYSTEM_NCNN will be turned off.")
        set(USE_SYSTEM_NCNN OFF)
    elseif(LITEOCR_ENABLE_VULKAN)
        # Heuristic: check whether the imported ncnn was built with Vulkan.
        get_target_property(_ncnn_iface_defs ncnn INTERFACE_COMPILE_DEFINITIONS)
        set(_ncnn_has_vulkan OFF)
        if(_ncnn_iface_defs)
            foreach(_def ${_ncnn_iface_defs})
                if(_def STREQUAL "NCNN_VULKAN")
                    set(_ncnn_has_vulkan ON)
                endif()
            endforeach()
        endif()
        if(NOT _ncnn_has_vulkan)
            message(WARNING "LITEOCR_ENABLE_VULKAN=ON but system ncnn does not appear to be built with Vulkan (NCNN_VULKAN not found in INTERFACE_COMPILE_DEFINITIONS). Falling back to bundled ncnn.")
            set(USE_SYSTEM_NCNN OFF)
        endif()
    endif()
endif()

if(NOT USE_SYSTEM_NCNN)
    # Build bundled ncnn from the git submodule.
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt")
        message(FATAL_ERROR
            "The ncnn submodule was not downloaded! "
            "Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    option(NCNN_INSTALL_SDK "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_VULKAN "" ${LITEOCR_ENABLE_VULKAN})
    # Do not let ncnn's bundled glslang register install rules; only LiteOCR
    # should be installed into the user's prefix.
    option(GLSLANG_ENABLE_INSTALL "" OFF)

    # Trim ncnn layer compilation to only those needed by LiteOCR models.
    # This is controlled entirely from LiteOCR; ncnn itself is left untouched.
    include(ncnn_layer_trim)
    liteocr_apply_ncnn_layer_trim()

    add_subdirectory(ncnn)
endif()
