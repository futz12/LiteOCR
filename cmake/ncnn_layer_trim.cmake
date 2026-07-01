# cmake/ncnn_layer_trim.cmake
# Trim ncnn layer compilation based on LiteOCR bundled models.
# All logic is implemented in CMake; no external tools are required.

option(LITEOCR_TRIM_NCNN_LAYERS "Only compile ncnn layers used by LiteOCR bundled models" ON)
option(LITEOCR_TRIM_NCNN_LAYERS_FROM_MODELS "Derive required ncnn layers from models/*.param" ON)

set(LITEOCR_NCNN_LAYERS "" CACHE STRING "Whitelist of ncnn layer names to compile")
set(LITEOCR_EXTRA_NCNN_LAYERS "" CACHE STRING "Extra ncnn layer names to keep in addition to the whitelist")

# Hardcoded base whitelist.
# - Layers from the current models/*.param analysis.
# - Internal dependencies used by ncnn core helpers in mat.cpp and by layer
#   implementations (e.g. Packing for convert_packing, Scale/Bias for
#   substract_mean_normalize, Cast for fp16/bf16/int8 conversions).
set(_LITEOCR_NCNN_LAYERS_DEFAULT
    batchnorm
    bias
    binaryop
    cast
    clip
    concat
    convolution
    convolutiondepthwise
    crop
    deconvolution
    dequantize
    expanddims
    flatten
    gelu
    gemm
    gridsample
    hardsigmoid
    hardswish
    innerproduct
    input
    interp
    layernorm
    memorydata
    mish
    multiheadattention
    noop
    packing
    padding
    permute
    pooling
    prelu
    quantize
    reduction
    relu
    requantize
    reshape
    scale
    shufflechannel
    sigmoid
    slice
    softmax
    split
    squeeze
    swish
    tanh
)

function(_liteocr_detect_layers_from_models out_var)
    set(_required_layers "noop")

    file(GLOB _param_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.param")
    foreach(_param IN LISTS _param_files)
        if(NOT EXISTS "${_param}")
            continue()
        endif()

        file(STRINGS "${_param}" _lines)
        list(LENGTH _lines _n)
        if(_n LESS_EQUAL 2)
            continue()
        endif()

        math(EXPR _last_idx "${_n} - 1")
        foreach(_idx RANGE 2 ${_last_idx})
            list(GET _lines ${_idx} _line)
            string(STRIP "${_line}" _line)
            if(_line STREQUAL "")
                continue()
            endif()

            # First token is the ncnn layer type name.
            string(REGEX REPLACE "^[ \t]*([A-Za-z0-9_]+).*" "\\1" _type "${_line}")
            if(_type STREQUAL "")
                continue()
            endif()

            string(TOLOWER "${_type}" _name)
            list(APPEND _required_layers "${_name}")
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES _required_layers)
    set(${out_var} "${_required_layers}" PARENT_SCOPE)
endfunction()

function(liteocr_apply_ncnn_layer_trim)
    # Parse all layers registered in ncnn/src/CMakeLists.txt.
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/CMakeLists.txt" _ncnn_layer_lines REGEX "^ncnn_add_layer\\(")
    set(_ncnn_all_layers "")
    foreach(_line IN LISTS _ncnn_layer_lines)
        string(REGEX REPLACE "ncnn_add_layer\\(([A-Za-z0-9_]+).*" "\\1" _class "${_line}")
        string(TOLOWER "${_class}" _name)
        list(APPEND _ncnn_all_layers "${_name}")
    endforeach()

    if(NOT LITEOCR_TRIM_NCNN_LAYERS)
        if(USE_SYSTEM_NCNN)
            return()
        endif()
        # Clear any stale WITH_LAYER_* cache entries from previous trimmed
        # configurations so that ncnn's option() can use its own defaults.
        foreach(_name IN LISTS _ncnn_all_layers)
            unset(WITH_LAYER_${_name} CACHE)
        endforeach()
        return()
    endif()

    if(USE_SYSTEM_NCNN)
        message(STATUS "LiteOCR: LITEOCR_TRIM_NCNN_LAYERS is ignored because USE_SYSTEM_NCNN is ON")
        return()
    endif()

    # Start from the hardcoded default whitelist (which covers internal
    # layer dependencies) and merge in any layers detected from models.
    set(_whitelist "${_LITEOCR_NCNN_LAYERS_DEFAULT}")
    if(LITEOCR_TRIM_NCNN_LAYERS_FROM_MODELS)
        _liteocr_detect_layers_from_models(_detected)
        list(APPEND _whitelist ${_detected})
        list(REMOVE_DUPLICATES _whitelist)
    endif()

    if(LITEOCR_EXTRA_NCNN_LAYERS)
        list(APPEND _whitelist ${LITEOCR_EXTRA_NCNN_LAYERS})
        list(REMOVE_DUPLICATES _whitelist)
    endif()

    set(LITEOCR_NCNN_LAYERS "${_whitelist}" CACHE STRING "Whitelist of ncnn layer names to compile" FORCE)
    message(STATUS "LiteOCR: trimming ncnn layers to ${_whitelist}")

    # Use CACHE FORCE so these values are visible inside the ncnn subdirectory.
    foreach(_name IN LISTS _ncnn_all_layers)
        if("${_name}" IN_LIST _whitelist)
            set(WITH_LAYER_${_name} ON CACHE BOOL "Build with layer ${_name}" FORCE)
        else()
            set(WITH_LAYER_${_name} OFF CACHE BOOL "Build with layer ${_name}" FORCE)
        endif()
    endforeach()
endfunction()
