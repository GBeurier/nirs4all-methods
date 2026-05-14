"""pls4all parity fixture generator (Python side).

Produces deterministic JSON fixtures conforming to
`parity/schema/fixture_schema_v1.json`. The generator is pinned to specific
versions of scikit-learn / NumPy (see ``requirements-lock.txt``) so the
output is bit-identical across runs.
"""

__version__ = "0.25.0"
