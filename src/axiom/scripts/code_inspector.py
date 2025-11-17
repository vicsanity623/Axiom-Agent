"""Inspects files for potential improvements."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

# Path is a runtime dependency for Typer, so we ignore the linter rule.
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Generator is a type-only dependency, so it goes in the type-checking block.
if TYPE_CHECKING:
    from collections.abc import Generator


# Configure a basic logger for this script
logger = logging.getLogger(__name__)


# --- Data Structure for Findings ---
@dataclass
class InspectionResult:
    """Represents a single issue found by the inspector."""

    line_number: int
    severity: str
    check_name: str
    message: str
    code_context: str = ""


# --- AST Visitor for Complex Checks ---
class CodeVisitor(ast.NodeVisitor):
    """An AST visitor that collects data for complex checks like complexity."""

    def __init__(self) -> None:
        self.complexity = 1
        self.function_defs: list[ast.FunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_defs.append(node)
        # We don't traverse into nested functions for this complexity calculation.

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.complexity += len(node.handlers)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.complexity += 1
        self.generic_visit(node)


# --- The Main Inspector Class ---
class CodeInspector:
    """
    A detailed code inspector that goes beyond standard linting to find
    common "code smells" and areas for improvement.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.source_code = file_path.read_text(encoding="utf-8")
        self.source_lines = self.source_code.splitlines()
        self.tree = ast.parse(self.source_code)

    def run_all_checks(self) -> list[InspectionResult]:
        """Runs all implemented checks and returns a list of results."""
        results = []
        for check_method in [
            self._check_line_length,
            self._check_docstrings,
            self._check_function_complexity_and_args,
            self._check_bad_practices,
            self._check_for_todos,
            self._check_deep_nesting,
            self._check_magic_values,
            self._check_god_objects,
            self._check_encapsulation_violations,
            self._check_old_string_formatting,
        ]:
            results.extend(list(check_method()))
        return sorted(results, key=lambda r: r.line_number)

    def _get_code_context(self, line_number: int, context_lines: int = 2) -> str:
        """Extracts a few lines of code around the issue for context."""
        start = max(0, line_number - context_lines - 1)
        end = min(len(self.source_lines), line_number + context_lines)
        context = []
        for i in range(start, end):
            prefix = "> " if (i + 1) == line_number else "  "
            context.append(f"{i + 1:4d} {prefix}{self.source_lines[i]}")
        return "\n".join(context)

    # --- Individual Check Methods ---

    def _check_line_length(self) -> Generator[InspectionResult, None, None]:
        """Checks for lines exceeding a recommended length."""
        for i, line in enumerate(self.source_lines, 1):
            if len(line) > 100:
                yield InspectionResult(
                    line_number=i,
                    severity="Style",
                    check_name="Line Too Long",
                    message=f"Line is {len(line)} characters long (recommended <= 100).",
                )

    def _check_docstrings(self) -> Generator[InspectionResult, None, None]:
        """Checks for missing or trivial docstrings in functions and classes."""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                line_no = node.lineno if hasattr(node, "lineno") else 1
                name = getattr(node, "name", "module")

                if not docstring:
                    if not (isinstance(node, ast.FunctionDef) and name.startswith("_")):
                        yield InspectionResult(
                            line_number=line_no,
                            severity="Documentation",
                            check_name="Missing Docstring",
                            message=f"Missing docstring for `{name}`.",
                        )
                elif len(docstring.split()) < 3:
                    yield InspectionResult(
                        line_number=line_no,
                        severity="Documentation",
                        check_name="Trivial Docstring",
                        message=f"Docstring for `{name}` is very short. Is it descriptive enough?",
                    )

    def _check_function_complexity_and_args(
        self,
    ) -> Generator[InspectionResult, None, None]:
        """Checks for overly complex functions and excessive arguments."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    yield InspectionResult(
                        line_number=node.lineno,
                        severity="Complexity",
                        check_name="Too Many Arguments",
                        message=f"Function `{node.name}` has {len(node.args.args)} arguments (recommended <= 5). Consider using a dataclass.",
                    )
                visitor = CodeVisitor()
                visitor.visit(node)
                if visitor.complexity > 10:
                    yield InspectionResult(
                        line_number=node.lineno,
                        severity="Complexity",
                        check_name="High Cyclomatic Complexity",
                        message=f"Function `{node.name}` has a complexity of {visitor.complexity} (recommended <= 10). Consider refactoring.",
                    )

    def _check_bad_practices(self) -> Generator[InspectionResult, None, None]:
        """Checks for common bad practices like bare `except`."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                yield InspectionResult(
                    line_number=node.lineno,
                    severity="Error Prone",
                    check_name="Bare Except",
                    message="Using a bare `except:` catches SystemExit and KeyboardInterrupt, making it hard to exit the program.",
                )
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.defaults:
                    if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                        yield InspectionResult(
                            line_number=node.lineno,
                            severity="Error Prone",
                            check_name="Mutable Default Argument",
                            message=f"Function `{node.name}` uses a mutable default argument. This can lead to unexpected behavior.",
                        )

    def _check_for_todos(self) -> Generator[InspectionResult, None, None]:
        """Scans for TODO/FIXME comments that may indicate unfinished work."""
        for i, line in enumerate(self.source_lines, 1):
            if "# TODO" in line or "# FIXME" in line:
                yield InspectionResult(
                    line_number=i,
                    severity="Maintenance",
                    check_name="Action Item Comment",
                    message=f"Found TODO/FIXME comment: {line.strip()}",
                )

    def _check_deep_nesting(self) -> Generator[InspectionResult, None, None]:
        """Checks for functions with deeply nested control flow."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                max_depth = self._get_max_nesting_depth(node)
                if max_depth > 4:
                    yield InspectionResult(
                        line_number=node.lineno,
                        severity="Complexity",
                        check_name="Deep Nesting",
                        message=f"Function `{node.name}` has a maximum nesting depth of {max_depth} (recommended <= 4).",
                    )

    def _get_max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Recursively calculates the maximum nesting depth of a node."""
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = current_depth
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth += 1
            max_depth = max(max_depth, self._get_max_nesting_depth(child, child_depth))
        return max_depth

    def _check_magic_values(self) -> Generator[InspectionResult, None, None]:
        """Checks for 'magic' strings or numbers used in logic."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Compare):
                for val in node.comparators + [node.left]:
                    if isinstance(val, ast.Constant) and isinstance(
                        val.value, (int, float)
                    ):
                        if val.value not in {0, 1, -1}:
                            yield InspectionResult(
                                line_number=node.lineno,
                                severity="Maintainability",
                                check_name="Magic Number",
                                message=f"Magic number `{val.value}` used in comparison. Consider defining it as a constant.",
                            )
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                if len(node.value.value.split()) > 1:
                    continue
                yield InspectionResult(
                    line_number=node.lineno,
                    severity="Maintainability",
                    check_name="Magic String",
                    message=f'Magic string `"{node.value.value}"` is returned. Consider using an Enum or a named constant.',
                )

    def _check_god_objects(self) -> Generator[InspectionResult, None, None]:
        """Checks for classes with a large number of methods (potential 'God Objects')."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(
                    1 for item in node.body if isinstance(item, ast.FunctionDef)
                )
                if method_count > 15:
                    yield InspectionResult(
                        line_number=node.lineno,
                        severity="Architecture",
                        check_name="Potential God Object",
                        message=f"Class `{node.name}` has {method_count} methods (recommended <= 15). It may have too many responsibilities.",
                    )

    def _check_encapsulation_violations(
        self,
    ) -> Generator[InspectionResult, None, None]:
        """Checks for direct access to 'private' attributes of other objects."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Attribute
            ):
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    yield InspectionResult(
                        line_number=node.lineno,
                        severity="Architecture",
                        check_name="Encapsulation Violation",
                        message=f"Direct access to a 'private' attribute `{node.attr}`. Consider using a public method instead.",
                    )

    def _check_old_string_formatting(self) -> Generator[InspectionResult, None, None]:
        """Checks for old '%' style string formatting in logging calls."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "%" in node.args[0].value
            ):
                yield InspectionResult(
                    line_number=node.lineno,
                    severity="Error Prone",
                    check_name="Old String Formatting",
                    message="Use of '%' for string formatting in a logger call. Use f-strings for safety and clarity.",
                )


# --- Typer CLI Application ---
app = typer.Typer(
    name="axiom-inspect",
    help="A detailed, opinionated code inspector for the Axiom Agent project.",
    add_completion=False,
)
console = Console()


@app.command()
def inspect(
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="The Python source file to inspect.",
    ),
) -> None:
    """
    Scans a single Python file for common code smells, complexity issues,
    and deviations from best practices.
    """
    console.rule(f"[bold cyan]Inspecting: {file_path.name}[/bold cyan]")

    try:
        inspector = CodeInspector(file_path)
        results = inspector.run_all_checks()
    except Exception as e:
        console.print(f"[bold red]Error parsing file:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not results:
        console.print(
            Panel(
                "[bold green]✅ No issues found. Excellent work![/bold green]",
                title="Inspection Complete",
                border_style="green",
            )
        )
        raise typer.Exit()

    table = Table(
        title=f"Found {len(results)} Issues in {file_path.name}", show_lines=True
    )
    table.add_column("Line", style="cyan", no_wrap=True)
    table.add_column("Severity", style="yellow")
    table.add_column("Check", style="magenta")
    table.add_column("Message & Code Context")

    for res in results:
        res.code_context = inspector._get_code_context(res.line_number)
        table.add_row(
            str(res.line_number),
            res.severity,
            res.check_name,
            f"{res.message}\n\n[dim]{res.code_context}[/dim]",
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold yellow]Inspection complete. Found {len(results)} potential areas for improvement.[/bold yellow]",
            border_style="yellow",
        )
    )


def main() -> None:
    """Main entry point for the script."""
    app()


if __name__ == "__main__":
    main()
