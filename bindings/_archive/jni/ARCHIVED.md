# Archived: JNI binding (frozen)

The JNI/Java binding is **on hold** (not among the current target languages:
R, Python, Octave/MATLAB, JS). It was moved here from `bindings/jni/` during
the donor/parity work because it still carries the pre-rename `p4a_*` naming
(`p4a_jni.c`, `io.github.pls4all`), which is a regression in the active tree
but allowed inside `_archive/` (see the root CLAUDE.md).

To reactivate (when the Java/Android target is picked up, cf. the `n4m-mobile`
subset): move it back under `bindings/jni/`, rename `p4a_*` → `n4m_*` and the
Java package `io.github.pls4all` → the n4m package, and wire it into the
binding raw-class generator + the conformance/parity gate.
