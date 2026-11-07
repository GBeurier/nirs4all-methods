# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#
# WASM build of libn4m via Emscripten. Produces (OUTPUT_NAME = n4m):
#   build/<preset>/bindings/js/n4m.js     — Emscripten glue (MODULARIZE, ES6)
#   build/<preset>/bindings/js/n4m.wasm   — WebAssembly module
# Plus the canonical TypeScript source under bindings/js/src/.
# WASM_BIGINT=1 stays ON: i64 args are marshalled as BigInt by the TS layer
# (see bindings/js/src/ffi.ts makeMatrixView).

if(NOT EMSCRIPTEN)
    message(WARNING "PLS4ALL_BUILD_BINDINGS_JS=ON but the active toolchain is "
                    "not Emscripten — skipping JS/WASM target. Use the "
                    "`emscripten` preset (it sets PLS4ALL_BUILD_BINDINGS_JS=ON "
                    "automatically).")
    return()
endif()

if(NOT TARGET n4m_c_static)
    message(FATAL_ERROR "n4m_c_static must be built for the JS binding.")
endif()

# Whitelist of public C ABI symbols we want to export to JS. The list is
# generated once from the canonical expected_symbols_linux.txt — every entry
# starts with `_` because Emscripten requires the underscore-prefixed C name.
set(_js_export_list "")
file(STRINGS "${PROJECT_SOURCE_DIR}/cpp/abi/expected_symbols_linux.txt"
     _exported_symbols)
foreach(sym IN LISTS _exported_symbols)
    if(sym MATCHES "^n4m_")
        list(APPEND _js_export_list "_${sym}")
    endif()
endforeach()
list(APPEND _js_export_list
    "_malloc" "_free"
    "_n4m_wasm_spectral_fit"
    "_n4m_wasm_spectral_transform"
    "_n4m_wasm_spectral_affine"
    "_n4m_wasm_pls_fit_legacy"
    "_n4m_wasm_pls_predict_from_coeffs"
    # Generic model dispatcher (Tier A algorithm-enum + Tier B coeff fits)
    "_n4m_wasm_model_fit"
    "_n4m_wasm_model_predict_from_coeffs"
    # AOM-PLS — operator-adaptive PLS (strict bank screen + SIMPLS on the winner)
    "_n4m_wasm_aom_fit"
    "_n4m_wasm_aom_chain_fit"
    # POP-PLS — per-component AOM (one operator per latent component)
    "_n4m_wasm_pop_fit"
    # AOM Ridge simplex blender + AOM operator-PLS stack (input-space coeffs)
    "_n4m_wasm_aom_ridge_fit"
    "_n4m_wasm_aom_stack_fit"
    # Splitters (KennardStone/SPXY/KMeans/KBinsStratified/SPlit/SystematicCircular)
    "_n4m_wasm_split"
    "_n4m_wasm_split_indices"
    "_n4m_wasm_splitter_indices"
    "_n4m_wasm_augmentation_apply"
    # Generic preprocessing dispatcher (raw-pointer surface; numerics in libn4m)
    "_n4m_wasm_pp_create"
    "_n4m_wasm_pp_fit"
    "_n4m_wasm_pp_transform"
    "_n4m_wasm_pp_state_len"
    "_n4m_wasm_pp_get_state"
    "_n4m_wasm_pp_set_state"
    "_n4m_wasm_pp_destroy")
string(REPLACE ";" "," _js_export_flag "${_js_export_list}")

add_executable(n4m_wasm
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_multimodal_layout.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_spectral.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_entry.c")
target_link_libraries(n4m_wasm PRIVATE n4m_c_static)
set_target_properties(n4m_wasm PROPERTIES
    OUTPUT_NAME "n4m"
    SUFFIX ".js")

target_link_options(n4m_wasm PRIVATE
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_ES6=1"
    "SHELL:-s ENVIRONMENT=web,worker,node"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s INITIAL_MEMORY=64MB"
    "SHELL:-s MAXIMUM_MEMORY=2GB"
    "SHELL:-s STACK_SIZE=8MB"
    "SHELL:-s WASM_BIGINT=1"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF64','HEAP32','HEAPU8','UTF8ToString','stringToUTF8','lengthBytesUTF8','getValue','setValue']"
    "SHELL:-s EXPORTED_FUNCTIONS=${_js_export_flag}"
    "SHELL:-s ASSERTIONS=1"
    -O2
)

# This branch contributes no target, options, output, or install changes in the
# default/native profiles. Qualification artifacts never replace package assets.
if(N4M_WASM_EH_MODEL STREQUAL "native-experimental")
    set(_n4m_eh_output "${CMAKE_CURRENT_BINARY_DIR}/native-eh-candidate")
    set(_n4m_eh_preflight "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_native_eh_preflight.js")
    set_target_properties(n4m_wasm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_n4m_eh_output}")
    target_link_options(n4m_wasm PRIVATE "SHELL:--pre-js ${_n4m_eh_preflight}")
    set_property(TARGET n4m_wasm APPEND PROPERTY LINK_DEPENDS "${_n4m_eh_preflight}")

    # The sole extra CXX object is an allocator fault fixture. Both modules link
    # the same owner archive (143 owner CXX objects); no guard or math is copied.
    add_executable(n4m_wasm_eh_boundary_tests EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_multimodal_layout.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_spectral.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_entry.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/native_eh_fault_injection.cpp")
    target_link_libraries(n4m_wasm_eh_boundary_tests PRIVATE n4m_c_static)
    target_compile_options(n4m_wasm_eh_boundary_tests PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-fwasm-exceptions>")
    set_target_properties(n4m_wasm_eh_boundary_tests PROPERTIES
        OUTPUT_NAME "n4m-eh-boundary-tests"
        SUFFIX ".js"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${_n4m_eh_output}")
    get_target_property(_n4m_eh_boundary_options n4m_wasm LINK_OPTIONS)
    # Keep every normal option, replacing only the test export list.
    list(FILTER _n4m_eh_boundary_options EXCLUDE REGEX "EXPORTED_FUNCTIONS=")
    set(_n4m_eh_test_exports ${_js_export_list}
        "_n4m_test_eh_arm" "_n4m_test_eh_allocations"
        "_n4m_test_eh_throws" "_n4m_test_eh_armed"
        "_n4m_test_eh_memory_snapshot")
    string(REPLACE ";" "," _n4m_eh_test_export_flag "${_n4m_eh_test_exports}")
    target_link_options(n4m_wasm_eh_boundary_tests PRIVATE
        ${_n4m_eh_boundary_options}
        "SHELL:-s EXPORTED_FUNCTIONS=${_n4m_eh_test_export_flag}")
    set_property(TARGET n4m_wasm_eh_boundary_tests APPEND PROPERTY
        LINK_DEPENDS "${_n4m_eh_preflight}")

    # Inline private build metadata, not an owner/scientific/state schema. No
    # additional product template is needed or installed. Pair/archive hashes
    # must be supplied and checked by the independent qualification launcher.
    foreach(_n4m_eh_kind IN ITEMS normal boundary)
        set(_n4m_eh_sidecar "n4m-runtime-capabilities.json")
        set(_n4m_eh_test_only false)
        if(_n4m_eh_kind STREQUAL "boundary")
            set(_n4m_eh_sidecar "n4m-eh-boundary-tests-capabilities.json")
            set(_n4m_eh_test_only true)
        endif()
        file(GENERATE OUTPUT "${_n4m_eh_output}/${_n4m_eh_sidecar}" CONTENT
"{\n  \"ehModel\": \"native-experimental\",\n  \"encoding\": \"wasm-legacy-eh\",\n  \"experimental\": true,\n  \"requiredFeature\": \"wasm-exception-handling\",\n  \"opcodes\": [\"try\", \"catch\", \"throw\", \"catch_all\", \"rethrow\", \"delegate\"],\n  \"traceStack\": true,\n  \"qualificationOnly\": ${_n4m_eh_test_only},\n  \"ownerArchive\": \"$<TARGET_FILE:n4m_c_static>\"\n}\n")
    endforeach()
else()
    # bindings/js is added before the top-level GNUInstallDirs include, so a first
    # configure would otherwise see an empty CMAKE_INSTALL_LIBDIR here.
    include(GNUInstallDirs)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/n4m.js"
        "${CMAKE_CURRENT_BINARY_DIR}/n4m.wasm"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/nirs4all-methods-js"
        OPTIONAL)
endif()
