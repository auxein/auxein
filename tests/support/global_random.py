"""An AST scan for global random state: the design forbids it anywhere in the package (§7.2, §8).

It flags
- numpy's legacy global random API, i.e. any public method of the module-level `RandomState` (`np.random.seed`, `rand`,
  `randn`, `uniform`, `normal`, `randint`, `choice`, `shuffle`, `permutation`, ...), however numpy or `numpy.random` is
  imported, and `from numpy.random import <legacy name>`;
- torch's global seeding and global generator state (`torch.manual_seed`, `torch.cuda.manual_seed_all`, ...), and torch's
  samplers (`torch.rand`, `torch.randn`, `torch.randint`, ...) called without an explicit `generator=`;
- the standard-library `random` module.

`numpy.random.Generator`, `PCG64`, `SeedSequence` and `default_rng` are fine: they are explicit, local state.
Torch is recognised when imported with `import torch` or obtained from `import_torch()`, which is how the package
imports it lazily.
"""

import ast
from pathlib import Path

import numpy as np

NUMPY_LEGACY = frozenset(name for name in dir(np.random.RandomState) if not name.startswith("_"))

TORCH_GLOBAL_STATE = frozenset(
    f"{module}.{name}"
    for module, names in {
        "torch": ("manual_seed", "seed", "initial_seed", "set_rng_state", "get_rng_state", "default_generator"),
        "torch.random": ("manual_seed", "seed", "initial_seed", "set_rng_state", "get_rng_state"),
        "torch.cuda": ("manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed", "set_rng_state", "set_rng_state_all"),
        "torch.mps": ("manual_seed", "seed", "set_rng_state"),
    }.items()
    for name in names
)

TORCH_SAMPLERS = frozenset(
    f"torch.{name}"
    for name in (
        "rand",
        "randn",
        "randint",
        "randperm",
        "multinomial",
        "normal",
        "bernoulli",
        "poisson",
        "rand_like",
        "randn_like",
        "randint_like",
    )
)


def _is_import_torch_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "import_torch") or (isinstance(func, ast.Attribute) and func.attr == "import_torch")


class _Scanner(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.aliases: dict[str, str] = {}  # local name -> dotted module path it stands for
        self.violations: list[str] = []

    def flag(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"{self.filename}:{getattr(node, 'lineno', 0)}: {message}")

    def dotted(self, node: ast.AST) -> str | None:
        """The dotted path an expression stands for (`np.random.seed` -> `numpy.random.seed`), if it is one."""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        if _is_import_torch_call(node):
            return "torch"
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "random":
                self.flag(node, "imports the standard-library `random` module")
            local = alias.asname or alias.name.split(".")[0]
            self.aliases[local] = alias.name if alias.asname else alias.name.split(".")[0]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level == 0 and module == "random":
            self.flag(node, "imports from the standard-library `random` module")
        if node.level == 0:
            for alias in node.names:
                path = f"{module}.{alias.name}"
                self.aliases[alias.asname or alias.name] = path
                if module == "numpy.random" and (alias.name in NUMPY_LEGACY or alias.name == "*"):
                    self.flag(node, f"imports numpy's legacy global random function `{alias.name}`")
                if path in TORCH_GLOBAL_STATE:
                    self.flag(node, f"imports torch's global random state `{path}`")

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_import_torch_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = "torch"
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = self.dotted(node)
        if path is not None:
            if path.startswith("numpy.random.") and path.count(".") == 2 and node.attr in NUMPY_LEGACY:
                self.flag(node, f"uses numpy's legacy global random API `{path}`")
            if path in TORCH_GLOBAL_STATE:
                self.flag(node, f"uses torch's global random state `{path}`")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = self.dotted(node.func)
        if path in TORCH_SAMPLERS and not any(keyword.arg == "generator" for keyword in node.keywords):
            self.flag(node, f"calls `{path}` without an explicit generator, so it draws from torch's global state")
        self.generic_visit(node)


def find_violations(source: str, filename: str = "<source>") -> list[str]:
    scanner = _Scanner(filename)
    scanner.visit(ast.parse(source, filename))
    return scanner.violations


def scan_package(path: Path) -> list[str]:
    """Violations in every Python file under `path`."""
    violations: list[str] = []
    for file in sorted(path.rglob("*.py")):
        violations += find_violations(file.read_text(), str(file))
    return violations
