# SPDX-License-Identifier: CECILL-2.1
#
# WASM build of libn4m via Emscripten. Produces:
#   build/<preset>/bindings/js/p4a.js     — Emscripten glue (MODULARIZE)
#   build/<preset>/bindings/js/p4a.wasm   — WebAssembly module
# Plus the canonical TypeScript source under bindings/js/src/.

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
    "_p4a_wasm_pls_fit_legacy"
    "_p4a_wasm_pls_predict_from_coeffs")
string(REPLACE ";" "," _js_export_flag "${_js_export_list}")

add_executable(pls4all_wasm
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm_entry.c")
target_link_libraries(pls4all_wasm PRIVATE n4m_c_static)
set_target_properties(pls4all_wasm PROPERTIES
    OUTPUT_NAME "p4a"
    SUFFIX ".js")

target_link_options(pls4all_wasm PRIVATE
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

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/p4a.js"
    "${CMAKE_CURRENT_BINARY_DIR}/p4a.wasm"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pls4all-wasm"
    OPTIONAL)
