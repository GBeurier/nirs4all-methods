include(GNUInstallDirs)
# Optional analysis engines are statically linked into the public C ABI.
# Default source builds need neither Rust nor third-party analysis headers.
foreach(_target IN ITEMS n4m_core n4m_c n4m_c_static)
    if(TARGET ${_target})
        target_compile_definitions(${_target} PRIVATE
            N4M_WITH_MIXEFF=$<BOOL:${N4M_WITH_MIXEFF}>
            N4M_WITH_RFR=$<BOOL:${N4M_WITH_RFR}>)
    endif()
endforeach()
if(N4M_WITH_RFR)
    target_include_directories(n4m_core SYSTEM PRIVATE
        "${PROJECT_SOURCE_DIR}/cpp/backends/rfr/include")
    # rfr includes cereal's JSON archive (never used by n4m), whose vendored RapidJSON
    # derives member iterators from the C++17-deprecated std::iterator (MSVC C4996/STL4015).
    # RapidJSON's own switch selects plain pointer iterators; vendored bytes stay unchanged.
    target_compile_definitions(n4m_core PRIVATE CEREAL_RAPIDJSON_NOMEMBERITERATORCLASS)
endif()
if(N4M_WITH_MIXEFF)
    find_program(N4M_CARGO_EXECUTABLE cargo REQUIRED)
    set(_rust_target_args)
    set(_rust_subdir "")
    if(EMSCRIPTEN)
        set(_rust_target_args --target wasm32-unknown-emscripten)
        set(_rust_subdir "wasm32-unknown-emscripten/")
    endif()
    if(CMAKE_CROSSCOMPILING AND NOT EMSCRIPTEN AND NOT APPLE)
        message(FATAL_ERROR "Mixed effects currently supports native desktop and Emscripten builds only; disable N4M_WITH_MIXEFF for this cross toolchain.")
    endif()
    if(APPLE AND CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(_rust_target_args --target aarch64-apple-darwin)
            set(_rust_subdir "aarch64-apple-darwin/")
        elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            set(_rust_target_args --target x86_64-apple-darwin)
            set(_rust_subdir "x86_64-apple-darwin/")
        else()
            message(FATAL_ERROR "Unsupported mixed-effects Apple architecture")
        endif()
    endif()
    if(MINGW)
        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
            message(FATAL_ERROR "Mixed effects MinGW builds require x86_64")
        endif()
        set(_rust_target_args --target x86_64-pc-windows-gnu)
        set(_rust_subdir "x86_64-pc-windows-gnu/")
    endif()
    if(WIN32 AND NOT MINGW)
        set(_rust_archive "n4m_mixeff_backend.lib")
    else()
        set(_rust_archive "libn4m_mixeff_backend.a")
    endif()
    set(N4M_MIXEFF_ARCHIVE_NAME "${_rust_archive}" PARENT_SCOPE)
    set(N4M_MIXEFF_ARCHIVE "${PROJECT_BINARY_DIR}/mixeff-cargo/${_rust_subdir}release/${_rust_archive}")
    set(_mixeff_manifest "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/Cargo.toml")
    set(_mixeff_sources "${_mixeff_manifest}"
        "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/Cargo.lock"
        "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/src/lib.rs")
    if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES ";")
        set(_mixeff_slices)
        foreach(_arch IN LISTS CMAKE_OSX_ARCHITECTURES)
            if(_arch STREQUAL "arm64")
                set(_rust_arch aarch64-apple-darwin)
            elseif(_arch STREQUAL "x86_64")
                set(_rust_arch x86_64-apple-darwin)
            else()
                message(FATAL_ERROR "Unsupported mixed-effects Apple architecture: ${_arch}")
            endif()
            set(_slice "${PROJECT_BINARY_DIR}/mixeff-cargo/${_rust_arch}/release/${_rust_archive}")
            add_custom_command(OUTPUT "${_slice}"
                COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${PROJECT_BINARY_DIR}/mixeff-cargo"
                    ${N4M_CARGO_EXECUTABLE} build --locked --release --target ${_rust_arch}
                    --manifest-path "${_mixeff_manifest}"
                DEPENDS ${_mixeff_sources} VERBATIM)
            list(APPEND _mixeff_slices "${_slice}")
        endforeach()
        find_program(N4M_LIPO_EXECUTABLE lipo REQUIRED)
        add_custom_command(OUTPUT "${N4M_MIXEFF_ARCHIVE}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/mixeff-cargo/release"
            COMMAND ${N4M_LIPO_EXECUTABLE} -create ${_mixeff_slices} -output "${N4M_MIXEFF_ARCHIVE}"
            DEPENDS ${_mixeff_slices} VERBATIM)
    else()
        add_custom_command(OUTPUT "${N4M_MIXEFF_ARCHIVE}"
            COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${PROJECT_BINARY_DIR}/mixeff-cargo"
                ${N4M_CARGO_EXECUTABLE} build --locked --release ${_rust_target_args}
                --manifest-path "${_mixeff_manifest}"
            DEPENDS ${_mixeff_sources}
            COMMENT "Building pinned mixeff backend without NLopt" VERBATIM)
    endif()
    add_custom_target(n4m_mixeff_build DEPENDS "${N4M_MIXEFF_ARCHIVE}")
    add_library(n4m_mixeff_backend STATIC IMPORTED GLOBAL)
    set_target_properties(n4m_mixeff_backend PROPERTIES IMPORTED_LOCATION "${N4M_MIXEFF_ARCHIVE}")
    add_dependencies(n4m_mixeff_backend n4m_mixeff_build)
    foreach(_target IN ITEMS n4m_c n4m_c_static)
        if(TARGET ${_target})
            target_link_libraries(${_target} PRIVATE
                $<BUILD_INTERFACE:n4m_mixeff_backend>
                $<INSTALL_INTERFACE:n4m::mixeff_backend>)
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                target_link_libraries(${_target} PRIVATE ${CMAKE_DL_LIBS} pthread m)
            elseif(WIN32)
                target_link_libraries(${_target} PRIVATE ws2_32 bcrypt userenv ntdll advapi32 dbghelp)
            endif()
        endif()
    endforeach()
    install(FILES "${N4M_MIXEFF_ARCHIVE}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(N4M_WITH_RFR)
    install(FILES "${PROJECT_SOURCE_DIR}/cpp/backends/rfr/LICENSE.txt"
        "${PROJECT_SOURCE_DIR}/cpp/backends/rfr/LICENSE_cereal-1.2.2"
        "${PROJECT_SOURCE_DIR}/cpp/backends/rfr/LICENSE_rapidjson"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/n4m/licenses/rfr)
endif()
if(N4M_WITH_MIXEFF)
    install(FILES "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/THIRD_PARTY_NOTICES.txt"
        "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/RUST_STANDARD_LIBRARY_NOTICES.html"
        "${PROJECT_SOURCE_DIR}/cpp/backends/mixeff/LICENSE.mixeff-rs"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/n4m/licenses/mixeff)
endif()

# Hidden Rust functions have individual sections. Drop unreachable sections
# after the shared-library export map has retained the public C ABI roots.
if(N4M_WITH_MIXEFF AND NOT EMSCRIPTEN AND TARGET n4m_c)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR MINGW)
        target_link_options(n4m_c PRIVATE -Wl,--gc-sections)
    elseif(APPLE)
        target_link_options(n4m_c PRIVATE -Wl,-dead_strip)
    endif()
endif()
