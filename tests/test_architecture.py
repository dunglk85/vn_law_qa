# tests/test_architecture.py
import ast
import os
import unittest
from pathlib import Path

# List of third-party concrete providers that must NOT leak into app/core/ or app/api.py
FORBIDDEN_PROVIDERS = {
    "openai",
    "cohere",
    "redis",
    "psycopg",
    "langchain_postgres",
    "langchain_openai",
    "langchain_cohere",
    "langchain_redis",
}


class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate project root (the directory containing 'app')
        cls.project_root = Path(__file__).parent.parent.resolve()
        cls.app_dir = cls.project_root / "app"

        # Find all Python files recursively under app/
        cls.python_files = []
        for root, _, files in os.walk(cls.app_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    cls.python_files.append(full_path)

    def _get_imports_and_classes(self, file_path: Path):
        """Parse file content and return list of resolved absolute import modules and class definitions."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")

        # Convert file_path's parent directory to a dot-separated module path relative to project_root
        rel_parent = file_path.parent.relative_to(self.project_root)
        folder_module = ".".join(rel_parent.parts)

        imports = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append((name.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                level = node.level
                module = node.module or ""
                if level == 0:
                    imports.append((module, node.lineno))
                else:
                    # Resolve relative import
                    base_parts = folder_module.split(".")
                    if level - 1 > 0:
                        base_parts = base_parts[:-(level - 1)]
                    if module:
                        resolved_parts = base_parts + module.split(".")
                    else:
                        resolved_parts = base_parts
                    imports.append((".".join(resolved_parts), node.lineno))

            elif isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        parts = []
                        curr = base
                        while isinstance(curr, ast.Attribute):
                            parts.append(curr.attr)
                            curr = curr.value
                        if isinstance(curr, ast.Name):
                            parts.append(curr.id)
                        parts.reverse()
                        bases.append(".".join(parts))
                classes.append({
                    "name": node.name,
                    "bases": bases,
                    "lineno": node.lineno
                })

        return imports, classes

    def test_ports_isolation(self):
        """Verify that Ports (app/ports/*) do not depend on Adapters, Core, Factory, or API."""
        ports_dir = self.app_dir / "ports"
        for file_path in self.python_files:
            if not file_path.is_relative_to(ports_dir):
                continue

            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                # Ports must not import adapters, core, factory, or api
                forbidden_prefixes = ["app.adapters", "app.core", "app.factory", "app.api"]
                for prefix in forbidden_prefixes:
                    if imp == prefix or imp.startswith(prefix + "."):
                        self.fail(
                            f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                            f"Port interface depends on {prefix} ('{imp}'). "
                            f"Ports must represent pure abstractions and be completely independent."
                        )

    def test_core_isolation(self):
        """Verify that Core business logic (app/core/*) does not depend on Adapters or Factory."""
        core_dir = self.app_dir / "core"
        for file_path in self.python_files:
            if not file_path.is_relative_to(core_dir):
                continue

            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                forbidden_prefixes = ["app.adapters", "app.factory", "app.api"]
                for prefix in forbidden_prefixes:
                    if imp == prefix or imp.startswith(prefix + "."):
                        self.fail(
                            f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                            f"Core service depends on {prefix} ('{imp}'). "
                            f"Core logic must only depend on abstract Ports or Config, and have its dependencies injected."
                        )

    def test_adapters_isolation(self):
        """Verify that Adapters (app/adapters/*) do not depend on Core logic or the API layer."""
        adapters_dir = self.app_dir / "adapters"
        for file_path in self.python_files:
            if not file_path.is_relative_to(adapters_dir):
                continue

            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                # Allow adapters to import A2A interface (pure abstract, no business logic)
                if imp == "app.core.a2a_client":
                    continue
                forbidden_prefixes = ["app.core", "app.api"]
                for prefix in forbidden_prefixes:
                    if imp == prefix or imp.startswith(prefix + "."):
                        self.fail(
                            f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                            f"Adapter depends on {prefix} ('{imp}'). "
                            f"Adapters implement Port interfaces and should not reference the business core or API bootstrap."
                        )

    def test_factory_isolation(self):
        """Verify that the Factory (app/factory.py) does not import Core services or API logic."""
        factory_file = self.app_dir / "factory.py"
        if not factory_file.exists():
            return

        imports, _ = self._get_imports_and_classes(factory_file)
        for imp, line in imports:
            forbidden_prefixes = ["app.core", "app.api"]
            for prefix in forbidden_prefixes:
                if imp == prefix or imp.startswith(prefix + "."):
                    self.fail(
                        f"Architectural Violation in factory.py:{line}\n"
                        f"Factory imports {prefix} ('{imp}'). "
                        f"The factory is responsible only for instantiating concrete adapters and should not depend on core services."
                    )

    def test_api_isolation(self):
        """Verify that the API layer (app/api.py) does not import adapters directly."""
        api_file = self.app_dir / "api.py"
        if not api_file.exists():
            return

        imports, _ = self._get_imports_and_classes(api_file)
        for imp, line in imports:
            forbidden_prefixes = ["app.adapters"]
            for prefix in forbidden_prefixes:
                if imp == prefix or imp.startswith(prefix + "."):
                    self.fail(
                        f"Architectural Violation in api.py:{line}\n"
                        f"API imports adapter '{imp}' directly. "
                        f"The API bootstrap must use the factory (app/factory.py) or dependency injection to resolve adapters."
                    )

    def test_no_concrete_providers_in_api_and_core(self):
        """Verify that API and Core do not import concrete third-party providers/SDKs directly."""
        restricted_paths = [self.app_dir / "core", self.app_dir / "api.py"]
        for file_path in self.python_files:
            is_restricted = any(file_path.is_relative_to(p) or file_path == p for p in restricted_paths)
            if not is_restricted:
                continue

            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                # Get the root package of the imported module
                root_pkg = imp.split(".")[0]
                if root_pkg in FORBIDDEN_PROVIDERS:
                    self.fail(
                        f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                        f"Direct import of concrete provider package '{imp}'. "
                        f"API and Core layers must use Port interfaces, and must not depend directly on concrete provider libraries."
                    )

    def test_adapters_implement_ports(self):
        """Verify that all classes defined in app/adapters/ ending with 'Adapter' subclass a class ending with 'Port'."""
        adapters_dir = self.app_dir / "adapters"
        for file_path in self.python_files:
            if not file_path.is_relative_to(adapters_dir):
                continue

            _, classes = self._get_imports_and_classes(file_path)
            for cls_info in classes:
                cls_name = cls_info["name"]
                if cls_name.endswith("Adapter"):
                    bases = cls_info["bases"]
                    # Check if at least one base class ends with "Port"
                    has_port_base = any(base.endswith("Port") for base in bases)
                    if not has_port_base:
                        self.fail(
                            f"Architectural Violation in {file_path.relative_to(self.project_root)}:{cls_info['lineno']}\n"
                            f"Adapter class '{cls_name}' does not inherit from any Port interface. "
                            f"All adapters must inherit from a Port class (e.g. VectorStorePort)."
                        )

    def test_agents_do_not_import_shared(self):
        """Verify that agents (app/agents/*) do not import from root-level shared module."""
        agents_dir = self.app_dir / "agents"
        for file_path in self.python_files:
            if not file_path.is_relative_to(agents_dir):
                continue
            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                if imp == "shared" or imp.startswith("shared."):
                    self.fail(
                        f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                        f"Agent imports 'shared' directly ('{imp}'). "
                        f"Agents must import from 'app.core.models' instead."
                    )

    def test_agents_do_not_import_adapters(self):
        """Verify that agents (app/agents/*) do not import adapters directly."""
        agents_dir = self.app_dir / "agents"
        for file_path in self.python_files:
            if not file_path.is_relative_to(agents_dir):
                continue
            imports, _ = self._get_imports_and_classes(file_path)
            for imp, line in imports:
                forbidden_prefixes = ["app.adapters"]
                for prefix in forbidden_prefixes:
                    if imp == prefix or imp.startswith(prefix + "."):
                        self.fail(
                            f"Architectural Violation in {file_path.relative_to(self.project_root)}:{line}\n"
                            f"Agent imports adapter '{imp}' directly. "
                            f"Agents must receive dependencies via constructor injection."
                        )


if __name__ == "__main__":
    unittest.main()
