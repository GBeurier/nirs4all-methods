---
orphan: true
---

# Android binding (archived)

The Android binding is **not a current target language**. It was never more
than a skeleton README and has been moved to `bindings/_archive/android/` — see
`bindings/_archive/android/ARCHIVED.md`.

The current target bindings are Python, R, MATLAB / Octave, and
JavaScript / WebAssembly. A real Android binding would be revived via the
`n4m-mobile` subset (a predict-only Java/Kotlin wrapper over the `n4m_c_static`
AAR, alongside the JNI binding in `bindings/_archive/jni/`). The CMake
`android-arm64` / `android-x86_64` presets that cross-compile the C static
library remain available.
