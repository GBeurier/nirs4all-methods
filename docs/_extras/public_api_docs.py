"""AST-backed public Python API metadata for generated documentation.

The documentation generator must never infer an import from an old binding
name or import ``n4m`` at build time: loading the package requires libn4m.
This module reads the source tree instead.  A name is public only when it is
listed in a module's ``__all__`` (or, for a module without ``__all__``, when it
is a non-private top-level definition or import).

``scan_public_api`` returns exports with their implementation kind and stable
source signature.  ``resolve_catalog_binding`` then validates the Python
binding declared by a catalog record, falling back only to an unambiguous,
same-name public symbol.  It deliberately returns an unresolved result rather
than inventing a convenient example.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


DEFAULT_N4M_SOURCE = (
    Path(__file__).resolve().parents[2] / "bindings" / "python" / "src" / "n4m"
)
DEFAULT_R_SOURCE = Path(__file__).resolve().parents[2] / "bindings" / "r" / "n4m"
DEFAULT_MATLAB_SOURCE = (
    Path(__file__).resolve().parents[2] / "bindings" / "matlab" / "+n4m"
)


@dataclass(frozen=True)
class PublicParameter:
    """A stable source-level callable parameter."""

    name: str
    annotation: str | None
    default: str | None
    kind: str

    @property
    def required(self) -> bool:
        return self.default is None and self.kind not in {"vararg", "varkw"}


@dataclass(frozen=True)
class PublicSymbol:
    """One source-verified public symbol.

    ``signature`` is the source-level call signature for functions and
    classes.  It is intentionally ``None`` for public constants and aliases
    whose implementation cannot be resolved statically.
    """

    module: str
    name: str
    kind: str
    signature: str | None
    source: str
    parameters: tuple[PublicParameter, ...] = ()
    source_line: int | None = None
    abi_symbols: tuple[str, ...] = ()

    @property
    def import_statement(self) -> str:
        return f"from {self.module} import {self.name}"

    @property
    def display_signature(self) -> str:
        return f"{self.name}{self.signature or '(...)'}"


@dataclass(frozen=True)
class BindingResolution:
    """Result of resolving a catalog/registry method to a public export."""

    status: str
    symbol: PublicSymbol | None = None
    requested_module: str | None = None
    requested_name: str | None = None
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified" and self.symbol is not None

    @property
    def import_statement(self) -> str | None:
        return self.symbol.import_statement if self.symbol else None

    @property
    def snippet(self) -> str | None:
        """An executable import-only snippet.

        Calling a function or constructing an estimator would require method-
        specific data and mandatory arguments.  Returning only the verified
        import avoids documentation examples that look runnable but are not.
        """
        return self.import_statement


@dataclass(frozen=True)
class CrossBindingSymbol:
    """A source-verified current R or MATLAB public callable."""

    language: str
    name: str
    kind: str
    signature: str | None
    source: str
    parameters: tuple[PublicParameter, ...] = ()


class CrossBindingIndex:
    """Public source index for one non-Python binding package."""

    def __init__(self, language: str, symbols: Iterable[CrossBindingSymbol]) -> None:
        self.language = language
        ordered = sorted(symbols, key=lambda item: (item.name, item.source))
        self._symbols = tuple(ordered)
        self._by_name = {item.name: item for item in ordered}

    @property
    def symbols(self) -> tuple[CrossBindingSymbol, ...]:
        return self._symbols

    def get(self, name: str) -> CrossBindingSymbol | None:
        return self._by_name.get(name)


class PublicApiIndex:
    """Lookup index returned by :func:`scan_public_api`."""

    def __init__(self, symbols: Iterable[PublicSymbol]) -> None:
        ordered = sorted(symbols, key=lambda item: (item.module, item.name))
        self._symbols = tuple(ordered)
        self._by_key = {(item.module, item.name): item for item in ordered}
        by_name: dict[str, list[PublicSymbol]] = defaultdict(list)
        by_normalized_name: dict[str, list[PublicSymbol]] = defaultdict(list)
        by_abi: dict[str, list[PublicSymbol]] = defaultdict(list)
        for item in ordered:
            by_name[item.name].append(item)
            if item.kind in {"class", "function"}:
                by_normalized_name[_normalized_name(item.name)].append(item)
                for abi_symbol in item.abi_symbols:
                    by_abi[abi_symbol].append(item)
        self._by_name = {name: tuple(items) for name, items in by_name.items()}
        self._by_normalized_name = {
            name: tuple(items) for name, items in by_normalized_name.items()
        }
        self._by_abi = {name: tuple(items) for name, items in by_abi.items()}

    @property
    def symbols(self) -> tuple[PublicSymbol, ...]:
        return self._symbols

    def get(self, module: str, name: str) -> PublicSymbol | None:
        return self._by_key.get((module, name))

    def named(self, name: str) -> tuple[PublicSymbol, ...]:
        return self._by_name.get(name, ())

    def normalized_named(self, name: str) -> tuple[PublicSymbol, ...]:
        """Return public callables whose name differs only by case/punctuation."""
        return self._by_normalized_name.get(_normalized_name(name), ())

    def abi_bound(self, abi_symbols: Iterable[str]) -> tuple[PublicSymbol, ...]:
        """Return public callables that reference one exact catalog C-ABI symbol."""
        found: dict[tuple[str, str], PublicSymbol] = {}
        for abi_symbol in abi_symbols:
            for item in self._by_abi.get(abi_symbol, ()):
                found[(item.module, item.name)] = item
        return tuple(found.values())


@dataclass(frozen=True)
class _ModuleSource:
    name: str
    path: Path
    is_package: bool
    tree: ast.Module


class _SourceResolver:
    """Resolve aliases through the source tree without importing n4m."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._modules: dict[str, _ModuleSource] = {}
        self._resolving: set[tuple[str, str]] = set()

    def module(self, name: str) -> _ModuleSource | None:
        if name in self._modules:
            return self._modules[name]
        if not name.startswith("n4m"):
            return None
        suffix = name.split(".")[1:]
        directory = self.root.joinpath(*suffix)
        candidates = ((directory / "__init__.py", True),
                      (directory.with_suffix(".py"), False))
        for path, is_package in candidates:
            if not path.is_file():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                return None
            source = _ModuleSource(name, path, is_package, tree)
            self._modules[name] = source
            return source
        return None

    @staticmethod
    def _literal_string_collection(node: ast.AST | None) -> set[str] | None:
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return None
        values: set[str] = set()
        for item in node.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                return None
            values.add(item.value)
        return values

    def exports(self, module_name: str) -> set[str]:
        module = self.module(module_name)
        if module is None:
            return set()
        declared_all: set[str] | None = None
        candidates: set[str] = set()
        for node in module.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    candidates.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    public_name = alias.asname or alias.name
                    if alias.name != "*" and not public_name.startswith("_"):
                        candidates.add(public_name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    public_name = alias.asname or alias.name.split(".")[0]
                    if not public_name.startswith("_"):
                        candidates.add(public_name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__all__":
                            parsed = self._literal_string_collection(value)
                            if parsed is not None:
                                declared_all = parsed
                        elif not target.id.startswith("_"):
                            candidates.add(target.id)
        return declared_all if declared_all is not None else candidates

    def _import_target(self, module: _ModuleSource, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        package = module.name if module.is_package else module.name.rpartition(".")[0]
        parts = package.split(".") if package else []
        up = node.level - 1
        if up > len(parts):
            return None
        base = parts[:len(parts) - up] if up else parts
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(part for part in base if part)

    def _alias_target(self, module: _ModuleSource,
                      name: str) -> tuple[str, str] | None:
        for node in module.tree.body:
            if isinstance(node, ast.ImportFrom):
                target_module = self._import_target(module, node)
                if not target_module:
                    continue
                for alias in node.names:
                    if alias.name != "*" and (alias.asname or alias.name) == name:
                        return target_module, alias.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name) or target.id != name:
                        continue
                    if isinstance(node.value, ast.Name):
                        return module.name, node.value.id
                    if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                        imported_module = self._imported_module(module, node.value.value.id)
                        if imported_module:
                            return imported_module, node.value.attr
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == name and node.value is not None:
                if isinstance(node.value, ast.Name):
                    return module.name, node.value.id
                if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                    imported_module = self._imported_module(module, node.value.value.id)
                    if imported_module:
                        return imported_module, node.value.attr
        return None

    def _imported_module(self, module: _ModuleSource, alias_name: str) -> str | None:
        for node in module.tree.body:
            if isinstance(node, ast.ImportFrom):
                target_module = self._import_target(module, node)
                if not target_module:
                    continue
                for alias in node.names:
                    if (alias.asname or alias.name) == alias_name and alias.name != "*":
                        # ``from package import child as alias`` can name a
                        # submodule or a symbol.  Only treat it as a module if
                        # the corresponding source module exists.
                        candidate = f"{target_module}.{alias.name}"
                        if self.module(candidate) is not None:
                            return candidate
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    public_name = alias.asname or alias.name.split(".")[0]
                    if public_name == alias_name and self.module(alias.name) is not None:
                        return alias.name
        return None

    def definition(self, module_name: str, name: str) -> tuple[ast.AST, _ModuleSource] | None:
        key = (module_name, name)
        if key in self._resolving:
            return None
        self._resolving.add(key)
        try:
            module = self.module(module_name)
            if module is None:
                return None
            for node in module.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and node.name == name:
                    return node, module
            target = self._alias_target(module, name)
            if target is None:
                return None
            return self.definition(*target)
        finally:
            self._resolving.remove(key)


def _default_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse exists on Python 3.11+
        return "..."


def _normalized_name(name: str) -> str:
    """Compare public spellings without guessing word substitutions."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _abi_symbols(node: ast.AST) -> tuple[str, ...]:
    """Extract literal C-ABI references from one resolved implementation AST."""
    return tuple(sorted(set(re.findall(
        r"\bn4m_[A-Za-z0-9_]+\b", ast.unparse(node)
    ))))


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str | None:
    """Format the callable surface, retaining ``/`` and bare ``*`` markers."""
    if isinstance(node, ast.ClassDef):
        init = next((item for item in node.body
                     if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
        if init is None:
            return "()"
        node = init
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults))
    defaults.extend(args.defaults)

    def render(arg: ast.arg, default: ast.AST | None, prefix: str = "") -> str:
        text = prefix + arg.arg
        if arg.annotation is not None:
            text += f": {_default_text(arg.annotation)}"
        if default is not None:
            text += f" = {_default_text(default)}"
        return text

    parts: list[str] = []
    positional_only: list[str] = []
    positional_or_keyword: list[str] = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if arg.arg in {"self", "cls"}:
            continue
        rendered = render(arg, default)
        if index < len(args.posonlyargs):
            positional_only.append(rendered)
        else:
            positional_or_keyword.append(rendered)
    parts.extend(positional_only)
    if positional_only:
        parts.append("/")
    parts.extend(positional_or_keyword)
    if args.vararg is not None:
        parts.append(render(args.vararg, None, "*"))
    elif args.kwonlyargs:
        parts.append("*")
    parts.extend(render(arg, default)
                 for arg, default in zip(args.kwonlyargs, args.kw_defaults))
    if args.kwarg is not None:
        parts.append(render(args.kwarg, None, "**"))
    return "(" + ", ".join(parts) + ")"


def _parameter_text(parameter: PublicParameter) -> str:
    prefix = "*" if parameter.kind == "vararg" else "**" if parameter.kind == "varkw" else ""
    text = prefix + parameter.name
    if parameter.annotation:
        text += f": {parameter.annotation}"
    if parameter.default is not None:
        text += f" = {parameter.default}"
    return text


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[PublicParameter, ...]:
    if isinstance(node, ast.ClassDef):
        init = next((item for item in node.body
                     if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
        if init is None:
            return ()
        node = init
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults))
    defaults.extend(args.defaults)
    parameters: list[PublicParameter] = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if arg.arg in {"self", "cls"}:
            continue
        parameters.append(PublicParameter(
            arg.arg,
            _default_text(arg.annotation) if arg.annotation is not None else None,
            _default_text(default) if default is not None else None,
            "positional_only" if index < len(args.posonlyargs) else "positional",
        ))
    if args.vararg is not None:
        parameters.append(PublicParameter(
            args.vararg.arg,
            _default_text(args.vararg.annotation) if args.vararg.annotation is not None else None,
            None,
            "vararg",
        ))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(PublicParameter(
            arg.arg,
            _default_text(arg.annotation) if arg.annotation is not None else None,
            _default_text(default) if default is not None else None,
            "keyword_only",
        ))
    if args.kwarg is not None:
        parameters.append(PublicParameter(
            args.kwarg.arg,
            _default_text(args.kwarg.annotation) if args.kwarg.annotation is not None else None,
            None,
            "varkw",
        ))
    return tuple(parameters)


def scan_public_api(source_dir: Path = DEFAULT_N4M_SOURCE) -> PublicApiIndex:
    """Read public ``n4m`` exports and source-level signatures by AST.

    The returned index contains every exported name across role modules.  An
    unresolved alias remains represented as ``kind='value'`` so callers can
    report a binding-coverage gap instead of claiming it is callable.
    """
    resolver = _SourceResolver(source_dir)
    symbol_rows: list[PublicSymbol] = []
    for path in sorted(source_dir.rglob("*.py")):
        # n4m.roles re-exposes catalog methods through the generated generic
        # role classes; their coverage is documented by
        # docs/parity/estimator_roles_coverage.md, not per-method pages.
        if "_impl" in path.parts or "roles" in path.parts:
            continue
        rel = path.relative_to(source_dir).with_suffix("")
        parts = ["n4m", *rel.parts]
        if parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join(parts)
        for name in sorted(resolver.exports(module_name)):
            resolved = resolver.definition(module_name, name)
            kind = "value"
            signature = None
            parameters: tuple[PublicParameter, ...] = ()
            source_line: int | None = None
            abi_symbols: tuple[str, ...] = ()
            source = path.relative_to(source_dir.parent).as_posix()
            if resolved is not None:
                node, origin = resolved
                source = origin.path.relative_to(source_dir.parent).as_posix()
                source_line = node.lineno
                if isinstance(node, ast.ClassDef):
                    kind = "class"
                    signature = _signature(node)
                    parameters = _parameters(node)
                    abi_symbols = _abi_symbols(node)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "function"
                    signature = _signature(node)
                    parameters = _parameters(node)
                    abi_symbols = _abi_symbols(node)
            symbol_rows.append(PublicSymbol(
                module_name, name, kind, signature, source, parameters,
                source_line, abi_symbols,
            ))
    return PublicApiIndex(symbol_rows)


def _balanced_parentheses(text: str, opening: int) -> tuple[str, int] | None:
    """Return text inside the parenthesis at ``opening`` without evaluating it."""
    if opening >= len(text) or text[opening] != "(":
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for offset, char in enumerate(text[opening:], start=opening):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1:offset], offset
    return None


def _split_signature_parameters(raw: str, language: str) -> tuple[PublicParameter, ...]:
    """Parse a lightweight source signature while retaining text defaults."""
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(raw + ","):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = raw[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    parameters: list[PublicParameter] = []
    for part in parts:
        if part in {"...", "varargin"}:
            parameters.append(PublicParameter(part, None, None, "varkw"))
            continue
        name, separator, default = part.partition("=")
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", name):
            continue
        parameters.append(PublicParameter(
            name,
            None,
            default.strip() if separator else None,
            "positional",
        ))
    return tuple(parameters)


def scan_r_public_api(source_dir: Path = DEFAULT_R_SOURCE) -> CrossBindingIndex:
    """Read the current R package's exported functions without running R."""
    namespace = source_dir / "NAMESPACE"
    try:
        namespace_text = namespace.read_text(encoding="utf-8")
    except OSError:
        return CrossBindingIndex("r", ())
    exported = set(re.findall(r"(?m)^\s*export\(\s*([A-Za-z][A-Za-z0-9_.]*)\s*\)",
                              namespace_text))
    definitions: dict[str, tuple[str, Path]] = {}
    for path in sorted((source_dir / "R").glob("*.R")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in re.finditer(
                r"(?m)^\s*([A-Za-z][A-Za-z0-9_.]*)\s*(?:<-|=)\s*function\s*(\()",
                text):
            parsed = _balanced_parentheses(text, match.start(2))
            if parsed is None:
                continue
            raw, _ = parsed
            definitions.setdefault(match.group(1), (raw, path))
    symbols: list[CrossBindingSymbol] = []
    repository = source_dir.parents[2]
    for name in sorted(exported):
        definition = definitions.get(name)
        if definition is None:
            continue
        raw, path = definition
        raw = re.sub(r"\s+", " ", raw).strip()
        parameters = _split_signature_parameters(raw, "r")
        symbols.append(CrossBindingSymbol(
            "r", name, "function", f"({raw.strip()})",
            path.relative_to(repository).as_posix(), parameters,
        ))
    return CrossBindingIndex("r", symbols)


def scan_matlab_public_api(source_dir: Path = DEFAULT_MATLAB_SOURCE) -> CrossBindingIndex:
    """Read public ``+n4m`` MATLAB/Octave entry points without running MATLAB."""
    symbols: list[CrossBindingSymbol] = []
    repository = source_dir.parents[2]
    function_pattern = re.compile(
        r"(?m)^\s*function\s+(?:\[[^\]]+\]\s*=\s*|[A-Za-z][A-Za-z0-9_]*\s*=\s*)?"
        r"([A-Za-z][A-Za-z0-9_]*)\s*(\()?"
    )
    for path in sorted(source_dir.glob("*.m")):
        if path.name == "Contents.m":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        class_match = re.search(r"(?m)^\s*classdef\s+([A-Za-z][A-Za-z0-9_]*)", text)
        if class_match:
            symbols.append(CrossBindingSymbol(
                "matlab", class_match.group(1), "class", "(...)",
                path.relative_to(repository).as_posix(), (),
            ))
            continue
        function_match = function_pattern.search(text)
        if function_match is None:
            continue
        name = function_match.group(1)
        opening = function_match.start(2) if function_match.group(2) else -1
        parsed = _balanced_parentheses(text, opening) if opening >= 0 else None
        raw = re.sub(r"\s+", " ", parsed[0]).strip() if parsed is not None else ""
        symbols.append(CrossBindingSymbol(
            "matlab", name, "function", f"({raw})",
            path.relative_to(repository).as_posix(),
            _split_signature_parameters(raw, "matlab"),
        ))
    return CrossBindingIndex("matlab", symbols)


CROSS_BINDING_OVERRIDES: dict[str, dict[str, str]] = {
    "models.pls.pls_fit_simple": {"r": "pls", "matlab": "pls_fit"},
    "preprocessing.derivatives.savitzky_golay": {"r": "savgol_transform", "matlab": "savgol_transform"},
    "preprocessing.scatter.snv": {"r": "snv_transform", "matlab": "snv_transform"},
    "splitters.kennard_stone": {"r": "kennard_stone_split", "matlab": "kennard_stone_split"},
}


def _cross_binding_candidates(method: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for value in (method.get("leaf"), method.get("registry_name")):
        if isinstance(value, str) and value:
            values.append(value)
    for legacy in method.get("legacy_ids", []) or []:
        if isinstance(legacy, str) and legacy:
            values.append(legacy.rsplit(".", 1)[-1])
    expanded: list[str] = []
    for value in values:
        expanded.extend((value, f"{value}_fit", f"{value}_select", f"{value}_transform"))
    return tuple(dict.fromkeys(expanded))


_CROSS_SAFE_ARGUMENTS = {
    "x": "X",
    "x_new": "X",
    "x_source": "X_source",
    "x_target": "X_target",
    "y": "Y",
    "y_source": "Y_source",
    "y_target": "Y_target",
}


def _cross_minimal_call(symbol: CrossBindingSymbol) -> str | None:
    arguments: list[str] = []
    for parameter in symbol.parameters:
        if not parameter.required:
            continue
        value = _CROSS_SAFE_ARGUMENTS.get(parameter.name.lower())
        if value is None:
            return None
        arguments.append(value)
    return f"{symbol.name}({', '.join(arguments)})"


def cross_bindings_for_catalog(
        method: Mapping[str, Any], r_public: CrossBindingIndex,
        matlab_public: CrossBindingIndex) -> dict[str, dict[str, Any]]:
    """Return source-verified R/MATLAB metadata for one catalog record.

    A call is emitted only when every required source argument is a conventional
    documented variable.  Otherwise callers receive the actual signature and
    package entry point rather than a made-up invocation.
    """
    method_id = str(method.get("method_id") or "")
    overrides = CROSS_BINDING_OVERRIDES.get(method_id, {})
    result: dict[str, dict[str, Any]] = {}
    for language, index in (("r", r_public), ("matlab", matlab_public)):
        candidate_names = [overrides[language]] if language in overrides \
            else list(_cross_binding_candidates(method))
        symbol = next((index.get(name) for name in candidate_names if index.get(name)), None)
        if symbol is None:
            continue
        call = _cross_minimal_call(symbol)
        if language == "r":
            snippet = "library(n4m)"
            if call:
                snippet += f"\nresult <- {call}"
        else:
            snippet = "addpath('bindings/matlab')"
            if call:
                snippet += f"\nresult = n4m.{call};"
        result[language] = {
            "symbol": symbol.name,
            "kind": symbol.kind,
            "signature": symbol.signature,
            "parameters": [
                {
                    "name": parameter.name,
                    "default": parameter.default,
                    "required": parameter.required,
                }
                for parameter in symbol.parameters
            ],
            "source": symbol.source,
            "snippet": snippet,
            "call": call,
        }
    return result


def _python_binding_fields(catalog_record: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not catalog_record:
        return None
    bindings = catalog_record.get("bindings")
    if not isinstance(bindings, Mapping):
        return None
    python = bindings.get("python")
    if not isinstance(python, Mapping):
        return None
    module = python.get("module")
    # The catalog historically calls this field ``class`` even for functions.
    name = python.get("class") or python.get("function") or python.get("symbol")
    if isinstance(module, str) and isinstance(name, str) and module and name:
        return module, name
    return None


def _conservative_candidates(catalog_record: Mapping[str, Any] | None,
                             registry_name: str | None) -> tuple[str, ...]:
    candidates: list[str] = []
    if catalog_record:
        leaf = catalog_record.get("leaf")
        if isinstance(leaf, str) and leaf:
            candidates.append(leaf)
        legacy = catalog_record.get("legacy_ids")
        if isinstance(legacy, list):
            for item in legacy:
                if isinstance(item, str) and item:
                    candidates.append(item.rsplit(".", 1)[-1])
    if registry_name:
        candidates.append(registry_name)
    return tuple(dict.fromkeys(candidates))


DOCUMENTED_PUBLIC_BINDINGS: dict[str, tuple[str, str]] = {
    # These spellings come from the scientific overlay's implementation
    # evidence.  They are catalog-name migrations only; every target still
    # has to resolve to a current AST public export below.
    "augmentation.drift.linear_drift": (
        "n4m.augmentation.drift", "LinearBaselineDrift"
    ),
    "augmentation.drift.path_length": (
        "n4m.augmentation.drift", "PathLengthAugmenter"
    ),
    "augmentation.drift.poly_drift": (
        "n4m.augmentation.drift", "PolynomialBaselineDrift"
    ),
    "augmentation.edge_artifacts.detector_rolloff": (
        "n4m.augmentation.instrument", "DetectorRollOffAugmenter"
    ),
    "augmentation.edge_artifacts.edge_artifacts": (
        "n4m.augmentation.instrument", "EdgeArtifactsAugmenter"
    ),
    "augmentation.edge_artifacts.edge_curvature": (
        "n4m.augmentation.instrument", "EdgeCurvatureAugmenter"
    ),
    "augmentation.edge_artifacts.stray_light": (
        "n4m.augmentation.instrument", "StrayLightAugmenter"
    ),
    "augmentation.edge_artifacts.truncated_peak": (
        "n4m.augmentation.instrument", "TruncatedPeakAugmenter"
    ),
    "augmentation.environmental.moisture": (
        "n4m.augmentation.instrument", "MoistureAugmenter"
    ),
    "augmentation.environmental.temperature": (
        "n4m.augmentation.instrument", "TemperatureAugmenter"
    ),
    "augmentation.mixup.local_mixup": (
        "n4m.augmentation.mixup", "LocalMixupAugmenter"
    ),
    "augmentation.mixup.mixup": (
        "n4m.augmentation.mixup", "MixupAugmenter"
    ),
    "augmentation.noise.gaussian_noise": (
        "n4m.augmentation.noise", "GaussianAdditiveNoise"
    ),
    "augmentation.noise.hetero_noise": (
        "n4m.augmentation.noise", "HeteroscedasticNoiseAugmenter"
    ),
    "augmentation.random.random_x_op": (
        "n4m.augmentation.mixup", "RandomXOperation"
    ),
    "augmentation.random.rotate_translate": (
        "n4m.augmentation.mixup", "RotateTranslateAugmenter"
    ),
    "augmentation.scattering.batch_effect": (
        "n4m.augmentation.scattering", "BatchEffectAugmenter"
    ),
    "augmentation.scattering.dead_band": (
        "n4m.augmentation.scattering", "DeadBandAugmenter"
    ),
    "augmentation.scattering.emsc_distort": (
        "n4m.augmentation.scattering", "EMSCDistortionAugmenter"
    ),
    "augmentation.scattering.instrument_broaden": (
        "n4m.augmentation.scattering", "InstrumentalBroadeningAugmenter"
    ),
    "augmentation.scattering.particle_size": (
        "n4m.augmentation.scattering", "ParticleSizeAugmenter"
    ),
    "augmentation.scattering.scatter_sim_msc": (
        "n4m.augmentation.scattering", "ScatterSimulationMSC"
    ),
    "augmentation.spectral.band_mask": (
        "n4m.augmentation.spectral", "BandMasking"
    ),
    "augmentation.spectral.band_perturb": (
        "n4m.augmentation.spectral", "BandPerturbationAugmenter"
    ),
    "augmentation.spectral.gauss_jitter": (
        "n4m.augmentation.spectral", "GaussianJitter"
    ),
    "augmentation.splines.spline_curve_simplification": (
        "n4m.augmentation.splines", "SplineCurveSimplificationAugmenter"
    ),
    "augmentation.splines.spline_smoothing": (
        "n4m.augmentation.splines", "SplineSmoothingAugmenter"
    ),
    "augmentation.splines.spline_x_perturbations": (
        "n4m.augmentation.splines", "SplineXPerturbationAugmenter"
    ),
    "augmentation.splines.spline_x_simplification": (
        "n4m.augmentation.splines", "SplineXSimplificationAugmenter"
    ),
    "augmentation.splines.spline_y_perturbations": (
        "n4m.augmentation.splines", "SplineYPerturbationAugmenter"
    ),
    "augmentation.wavelength.local_warp": (
        "n4m.augmentation.wavelength", "LocalWarpAugmenter"
    ),
    "preprocessing.alignment.cow_align": (
        "n4m.transform.alignment", "CorrelationOptimizedWarping"
    ),
    "preprocessing.alignment.dtw_align": (
        "n4m.transform.alignment", "DynamicTimeWarpingAlignment"
    ),
    "preprocessing.alignment.icoshift_align": (
        "n4m.transform.alignment", "IcoshiftAlignment"
    ),
    "preprocessing.alignment.xcorr_align": (
        "n4m.transform.alignment", "CrossCorrelationAlignment"
    ),
    "preprocessing.baselines.saps": (
        "n4m.domain_adaptation.standardization",
        "ScoreAugmentedProjectionStandardization",
    ),
    "preprocessing.transfer.slope_bias": (
        "n4m.domain_adaptation.standardization", "SlopeBiasCorrection"
    ),
    "selection.interval": (
        "n4m.feature_selection.interval", "IntervalGenerator"
    ),
    "selection.wvc_threshold": (
        "n4m.feature_selection.wrapper", "WVCThreshold"
    ),
    "splitters.split_splitter": (
        "n4m.model_selection.splitters", "DataTwinning"
    ),
    "diagnostics.regression_metrics": (
        "n4m.metrics.scoring", "nirs_metrics"
    ),
}


def _distinct_symbols(symbols: Iterable[PublicSymbol]) -> tuple[PublicSymbol, ...]:
    unique = {(symbol.module, symbol.name): symbol for symbol in symbols}
    return tuple(unique.values())


def _preferred_symbol(symbols: Iterable[PublicSymbol],
                      catalog_record: Mapping[str, Any] | None) -> PublicSymbol:
    """Choose a canonical spelling among public aliases of one implementation."""
    leaf = str(catalog_record.get("leaf") or "") if catalog_record else ""
    normalized_leaf = _normalized_name(leaf)
    return min(
        symbols,
        key=lambda symbol: (
            _normalized_name(symbol.name) != normalized_leaf,
            symbol.kind != "class",
            len(symbol.name),
            symbol.module,
            symbol.name,
        ),
    )


def resolve_catalog_binding(index: PublicApiIndex,
                            catalog_record: Mapping[str, Any] | None,
                            registry_name: str | None = None) -> BindingResolution:
    """Resolve one catalog/registry method to a validated public Python name.

    A declared catalog module/name wins and must be a callable public export.
    For older records with no declaration, it accepts only three source-backed
    migrations: a documented role-package spelling, an unambiguous difference
    in case/punctuation, or a public wrapper linked to the exact C-ABI symbol.
    It never derives a class name from a broad fuzzy match.
    """
    declared = _python_binding_fields(catalog_record)
    if declared:
        module, name = declared
        symbol = index.get(module, name)
        if symbol is not None and symbol.kind in {"class", "function"}:
            return BindingResolution("verified", symbol, module, name)
        if symbol is not None:
            return BindingResolution(
                "declared_but_not_callable", symbol, module, name,
                "catalog Python binding resolves to a public module/value, not a function or class",
            )
        return BindingResolution(
            "declared_but_not_public", None, module, name,
            "catalog Python binding is absent from the current public n4m AST surface",
        )

    method_id = str(catalog_record.get("method_id") or "") if catalog_record else ""
    documented = DOCUMENTED_PUBLIC_BINDINGS.get(method_id)
    if documented:
        module, name = documented
        symbol = index.get(module, name)
        if symbol is not None and symbol.kind in {"class", "function"}:
            return BindingResolution("verified", symbol, module, name)
        return BindingResolution(
            "documented_binding_not_public", None, module, name,
            "scientific-overlay binding is absent from the current public n4m AST surface",
        )

    candidates = _conservative_candidates(catalog_record, registry_name)
    exact = _distinct_symbols(
        symbol for candidate in candidates for symbol in index.named(candidate)
        if symbol.kind in {"class", "function"}
    )
    if len(exact) == 1:
        symbol = exact[0]
        return BindingResolution("verified", symbol, symbol.module, symbol.name)
    if len(exact) > 1:
        return BindingResolution(
            "ambiguous", detail="multiple public exact-name exports; catalog binding is required"
        )

    normalized = _distinct_symbols(
        symbol for candidate in candidates for symbol in index.normalized_named(candidate)
    )
    if len(normalized) == 1:
        symbol = normalized[0]
        return BindingResolution("verified", symbol, symbol.module, symbol.name)
    if len(normalized) > 1:
        return BindingResolution(
            "ambiguous", detail="multiple public normalized-name exports; catalog binding is required"
        )

    abi_symbols = (
        catalog_record.get("c_surface", catalog_record.get("abi_symbols", ()))
        if catalog_record else ()
    )
    if isinstance(abi_symbols, str):
        abi_symbols = () if abi_symbols == "none" else (abi_symbols,)
    if isinstance(abi_symbols, (list, tuple)):
        abi_candidates = index.abi_bound(
            item for item in abi_symbols if isinstance(item, str)
        )
    else:
        abi_candidates = ()
    by_origin: dict[tuple[str, int | None], list[PublicSymbol]] = defaultdict(list)
    for symbol in abi_candidates:
        by_origin[(symbol.source, symbol.source_line)].append(symbol)
    if len(by_origin) == 1:
        symbol = _preferred_symbol(next(iter(by_origin.values())), catalog_record)
        return BindingResolution("verified", symbol, symbol.module, symbol.name)
    if len(by_origin) > 1:
        return BindingResolution(
            "ambiguous", detail="multiple public wrappers reference catalog C-ABI symbols"
        )

    return BindingResolution(
        "not_declared",
        detail="no current public Python binding is declared or proven by source",
    )


_SAFE_EXAMPLE_ARGUMENTS = {
    "x": "X",
    "x_train": "X_train",
    "xtrain": "X_train",
    "y": "y",
    "y_train": "y_train",
    "ytrain": "y_train",
}


def _minimal_function_call(symbol: PublicSymbol) -> str | None:
    """Return a safe call using conventional ``X``/``y`` variables, if exact.

    Required values such as a context, folds, target-domain matrix, or tuning
    grid cannot be safely fabricated.  In those cases callers receive the
    verified import and the full signature but no pseudo-runnable call.
    """
    if symbol.kind != "function":
        return None
    arguments: list[str] = []
    for parameter in symbol.parameters:
        if not parameter.required:
            continue
        value = _SAFE_EXAMPLE_ARGUMENTS.get(parameter.name.lower())
        if value is None:
            return None
        if parameter.kind == "keyword_only":
            arguments.append(f"{parameter.name}={value}")
        else:
            arguments.append(value)
    return f"{symbol.name}({', '.join(arguments)})"


def binding_for_catalog(method: Mapping[str, Any], public: PublicApiIndex,
                        registry_name: str | None = None) -> dict[str, Any] | None:
    """Return verified documentation metadata for one catalog method.

    The result is intentionally plain data so the Markdown generator can use
    it without importing a binding package.  ``call`` is present only for a
    function whose every required input is one of the conventional example
    variables ``X``, ``X_train``, ``y`` or ``y_train``; otherwise the caller
    should render the import and signature alone.
    """
    registry_name = registry_name or str(method.get("registry_name") or "") or None
    resolution = resolve_catalog_binding(public, method, registry_name)
    if not resolution.verified or resolution.symbol is None:
        return None
    symbol = resolution.symbol
    params = [
        {
            "name": parameter.name,
            "annotation": parameter.annotation,
            "default": parameter.default,
            "kind": parameter.kind,
            "required": parameter.required,
        }
        for parameter in symbol.parameters
    ]
    return {
        "module": symbol.module,
        "symbol": symbol.name,
        "kind": symbol.kind,
        "signature": symbol.signature,
        "parameters": params,
        "source": symbol.source,
        "source_line": symbol.source_line,
        "abi_symbols": list(symbol.abi_symbols),
        "import": symbol.import_statement,
        "call": _minimal_function_call(symbol),
    }


__all__ = [
    "BindingResolution",
    "binding_for_catalog",
    "CrossBindingIndex",
    "CrossBindingSymbol",
    "cross_bindings_for_catalog",
    "CROSS_BINDING_OVERRIDES",
    "DEFAULT_N4M_SOURCE",
    "DEFAULT_MATLAB_SOURCE",
    "DEFAULT_R_SOURCE",
    "DOCUMENTED_PUBLIC_BINDINGS",
    "PublicApiIndex",
    "PublicParameter",
    "PublicSymbol",
    "resolve_catalog_binding",
    "scan_matlab_public_api",
    "scan_public_api",
    "scan_r_public_api",
]
