# tests/test_introspection_suite.py

import sys
from pathlib import Path

import pytest

# Ensure src is in sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.metacognitive_engine import SandboxVerifier


@pytest.fixture
def dummy_project(tmp_path: Path) -> Path:
    """Creates a miniature, self-contained project structure for testing."""
    project_root = tmp_path / "dummy_project"

    # Create src directory and a simple module
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "dummy_module.py").write_text(
        "def calculate(a, b):\n"
        '    """A simple function to be tested."""\n'
        "    return a + b\n",
    )

    # Create tests directory and a simple test
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_dummy.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))\n"
        "from dummy_module import calculate\n\n"
        "def test_calculate():\n"
        "    assert calculate(2, 3) == 5\n",
    )

    # Add a basic pyproject.toml for pytest to recognize the root
    (project_root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nminversion = "6.0"\n',
    )

    # FIX: Create a dummy check.sh script that always succeeds.
    # This satisfies the new requirement of the SandboxVerifier.
    check_script = project_root / "check.sh"
    check_script.write_text(
        "#!/bin/bash\n"
        "echo 'Running dummy check.sh...'\n"
        "exit 0\n",  # Exit with a success code
    )
    check_script.chmod(0o755)  # Make the script executable

    return project_root


def test_sandbox_verifier_accepts_safe_change(dummy_project: Path, monkeypatch):
    """
    Given a valid code change that passes the test suite,
    the SandboxVerifier should return True.
    """
    # Arrange
    monkeypatch.chdir(dummy_project)

    verifier = SandboxVerifier()

    original_file_path = Path("src/dummy_module.py")
    function_name = "calculate"

    safe_new_code = (
        "def calculate(a, b):\n"
        '    """A simple function to be tested."""\n'
        "    # This is a safe change\n"
        "    return a + b\n"
    )

    # Act
    is_safe = verifier.verify_change(original_file_path, function_name, safe_new_code)

    # Assert
    assert is_safe is True


def test_sandbox_verifier_rejects_unsafe_change(dummy_project: Path, monkeypatch):
    """
    Given a breaking code change that fails the test suite,
    the SandboxVerifier should return False.
    """
    # Arrange
    monkeypatch.chdir(dummy_project)

    # Modify the check.sh for this specific test to fail if pytest fails.
    # This makes the test more realistic.
    check_script = dummy_project / "check.sh"
    check_script.write_text(
        "#!/bin/bash\n"
        "echo 'Running dummy check.sh with pytest...'\n"
        "pytest\n",  # This will exit with a non-zero code if tests fail
    )
    check_script.chmod(0o755)

    verifier = SandboxVerifier()

    original_file_path = Path("src/dummy_module.py")
    function_name = "calculate"

    unsafe_new_code = (
        "def calculate(a, b):\n"
        '    """A simple function to be tested."""\n'
        "    return a - b  # This will fail the assertion `assert 5 == -1`\n"
    )

    # Act
    is_safe = verifier.verify_change(original_file_path, function_name, unsafe_new_code)

    # Assert
    assert is_safe is False
