import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from axiom.metacognitive_engine import SandboxVerifier


@pytest.fixture
def dummy_project(tmp_path: Path) -> Path:
    """Creates a miniature, self-contained project structure for testing."""
    project_root = tmp_path / "dummy_project"

    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "dummy_module.py").write_text(
        "def calculate(a, b):\n"
        '    """A simple function to be tested."""\n'
        "    return a + b\n",
    )

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

    (project_root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nminversion = "6.0"\n',
    )

    check_script = project_root / "check.sh"
    check_script.write_text(
        "#!/bin/bash\necho 'Running dummy check.sh...'\nexit 0\n",
    )
    check_script.chmod(0o755)

    return project_root


def test_sandbox_verifier_accepts_safe_change(dummy_project: Path, monkeypatch):
    """
    Given a valid code change that passes the test suite,
    the SandboxVerifier should return True.
    """
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

    is_safe = verifier.verify_change(original_file_path, function_name, safe_new_code)

    assert is_safe is True


def test_sandbox_verifier_rejects_unsafe_change(dummy_project: Path, monkeypatch):
    """
    Given a breaking code change that fails the test suite,
    the SandboxVerifier should return False.
    """
    monkeypatch.chdir(dummy_project)

    check_script = dummy_project / "check.sh"
    check_script.write_text(
        "#!/bin/bash\necho 'Running dummy check.sh with pytest...'\npytest\n",
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

    is_safe = verifier.verify_change(original_file_path, function_name, unsafe_new_code)

    assert is_safe is False
