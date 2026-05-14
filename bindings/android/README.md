# Android binding (skeleton)

Phase 0 reserves this directory. Phase 2 ships an AAR that bundles the
predict-only static library with a Kotlin wrapper.

ABI targets: **arm64-v8a** and **x86_64** (the emulator). 32-bit ABIs
(`armeabi-v7a`) are explicitly excluded per roadmap §10 because the
`p4a_matrix_view_t` struct layout depends on LP64/LLP64.

Planned layout:

```
bindings/android/
├── settings.gradle.kts
└── pls4all-android/
    ├── build.gradle.kts
    ├── src/main/cpp/
    │   ├── CMakeLists.txt    # add_subdirectory(...) onto cpp/src in this repo
    │   └── jni_gateway.cpp   # JNI wrapper around the C ABI
    └── src/main/kotlin/io/pls4all/
        └── Pls4all.kt        # Phase 2: Context, Config, Model
```
