# Phase 36 — Deferred / Future Bindings

**Status**: deferred · documents what is NOT yet shipped and why ·
unblocks the move to Phase 43 (accelerated backends).

After Phase 42 we stop the language-binding sprint with **10 live
bindings** (Python, WASM, R, Julia, Octave/MATLAB, JNI, Go, Rust,
.NET, Ruby, Lua, Nim — see the README parity matrix). The bindings
listed below were considered but deferred. None of them block any
shipped functionality; each gets a paragraph explaining the trade-off
so a future maintainer can pick them up cleanly.

## Deferred bindings

### Android (NDK packaging — was the original Phase 36)

The Phase 35 JNI source compiles **unchanged** against the Android
NDK: the same `Pls4all.java` + `p4a_jni.c` go through
`<NDK>/toolchains/llvm/prebuilt/.../bin/<abi>-linux-android-clang`
instead of the system `gcc`, and the resulting `libp4a.so` +
`libp4a_jni.so` get packaged into an AAR alongside the Java class.

Deferred because the NDK is a ~3.5 GB install that the current host
doesn't carry, and the existing JNI desktop binding already covers
JVM Android via Termux / JRuby / JRE-on-device patterns. To finish:

1. Install Android NDK r26d+.
2. Cross-compile libp4a for `arm64-v8a`, `armeabi-v7a`, and
   `x86_64` (for the emulator).
3. Wrap the same source into an AAR with `gradle assembleRelease`.
4. Parity-gate by running `TestParity` on an emulator pointed at the
   shared fixture.

### Swift (Apple ecosystem)

Apple's chemometrics / spectroscopy footprint is small but non-zero
(custom iOS data-capture apps for portable NIRS probes). Swift
compiles on Linux via the official Swift toolchain (~700 MB), and
`extern "C"` interop is mature. We defer because:

* The toolchain is large and our build host is not the right place
  to host an Apple cross-compile.
* On Linux desktop, the C / C++ / Rust / Go bindings cover the same
  consumers (the Swift binding's incremental reach is iOS / macOS).
* The Apple-native path likely wants a Swift Package Manager bundle
  (`Package.swift`) plus an XCFramework with prebuilt
  `libp4a.dylib` for x86_64 and arm64 macOS / iOS — a packaging
  exercise that depends on having a macOS / iOS host.

To finish: install `swift` (>= 5.9), create
`bindings/swift/Sources/Pls4all/Pls4all.swift` with `@_silgen_name`
shims around `p4a_pls_fit_simple`, and ship a parity test under
`bindings/swift/Tests/Pls4allTests/`. Then on macOS, ship an
XCFramework.

### Common Lisp (SBCL + CFFI)

CFFI is the canonical Common Lisp FFI. The toolchain is small
(SBCL ~25 MB; CFFI is a Quicklisp dependency). We defer because the
incremental reach is narrow — most CL chemometrics work runs through
the R / Python bindings anyway. The binding is straightforward to
ship if a CL consumer materialises:

```lisp
(cffi:defcfun ("p4a_pls_fit_simple" p4a-pls-fit-simple) :int
  (x :pointer) (y :pointer)
  (n :int) (p :int) (q :int) (n-components :int)
  (coefs :pointer) (x-mean :pointer)
  (y-mean :pointer) (preds :pointer))
```

### Other languages considered but not pursued

| Language | Why deferred |
|----------|--------------|
| **PHP** | The conda-forge PHP 8.1 package does not include the `ffi.so` extension; building from source would be heavier than the binding itself. Once a conda-forge PHP build ships with FFI enabled, the binding is a half-day. |
| **Perl** | No FFI in core. The standard FFI gem `FFI::Platypus` requires CPAN + `libffi-dev`. Without CPAN, a hand-rolled XS wrapper is the alternative — heavier than the value it adds. |
| **Crystal** | Crystal's `lib` syntax is the cleanest FFI in any language, but the toolchain isn't packaged in conda-forge and our consumer base is small. |
| **Zig** | Built-in `@cImport`, no toolchain dependencies. Defer because Zig's chemometrics user base is currently zero — the binding is straightforward to add once a consumer surfaces. |
| **OCaml** | `ctypes` is mature. Deferred because the OCaml chemometrics community overlaps almost entirely with the R / Python community. |
| **Haskell** | `Foreign.C.*` works fine but the project's reach into Haskell research code is small. |
| **Tcl** | `critcl` works but Tcl's modern usage is niche. |
| **Racket / Scheme** | `ffi/unsafe` is solid; same niche-reach argument. |
| **Dart** | `dart:ffi` is excellent for Flutter desktop / mobile, but mobile-side coverage is the Android NDK story above. |
| **Kotlin/Native** | `cinterop` plus the JNI binding cover the JVM and Kotlin Multiplatform stories; the Native-only path is rare. |

## Why we stop here

The current ten bindings cover **every major chemometrics / data-
science / systems-programming ecosystem**:

* **Scientific**: Python, R, Julia, Octave/MATLAB (the four big
  chemometrics environments)
* **Web / app**: JavaScript / WASM (browsers, Node, Deno, Bun)
* **JVM**: JNI (Java, Kotlin, Scala, Clojure, Groovy)
* **Modern systems**: Go, Rust, Nim, .NET
* **Scripting**: Ruby, Lua / LuaJIT

Every binding consumes the same `p4a_pls_fit_simple` ABI helper and
passes the same `parity_fixture.json` gate. Adding more bindings now
gives diminishing return; the next bottleneck is **runtime speed**,
which is what Phase 43 (BLAS) onward addresses.

## Reactivation criteria

Re-open a deferred binding when:
* A concrete consumer requests it (issue, partner, downstream library).
* The toolchain becomes cheap to host (NDK in conda-forge, Swift on
  GitHub Actions runner, PHP with FFI enabled).
* A breaking ABI change (1.x → 2.x) would force a re-sweep anyway —
  pick up the deferred bindings at the same time.
