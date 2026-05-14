# MATLAB binding (skeleton)

Phase 0 reserves this directory. Phase 2 ships a MEX-based binding that
provides a MATLAB class wrapper around `libp4a`.

Planned layout:

```
bindings/matlab/
├── +pls4all/
│   ├── version.m
│   ├── Context.m
│   ├── Config.m
│   └── Model.m              # Phase 2
├── mex/
│   ├── p4a_version_mex.cpp
│   ├── p4a_context_mex.cpp
│   ├── p4a_model_mex.cpp    # Phase 2
│   └── build_mex.m
└── tests/
    └── test_version.m
```

The MEX gateway uses the modern C++ MEX API and dynamically loads the
shared library, so the toolbox can ship as a single `.mltbx` archive.
