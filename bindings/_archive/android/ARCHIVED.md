# Archived: Android binding (stub, frozen)

The Android binding was never more than a placeholder (`README.md` only) and
Android is not among the current target languages (R, Python, Octave/MATLAB, JS;
see `bindings/SPEC.md`). It was moved here from `bindings/android/` so the active
`bindings/` tree contains only the four target languages.

Note: the **CMake** `android-arm64` / `android-x86_64` presets in
`CMakePresets.json` are unrelated to this directory — they cross-compile the C
ABI static library (`n4m_c_static`) for a predict-only AAR and remain active.
This archived directory only ever held the (empty) Java/Kotlin binding stub.

To reactivate (cf. the `n4m-mobile` subset): build the Java/Kotlin wrapper over
the `n4m_c_static` AAR under `bindings/android/`, alongside the JNI binding in
`bindings/_archive/jni/`.
