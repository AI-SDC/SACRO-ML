"""Guard against introducing OpenML network dependency into tests."""

from pathlib import Path


def test_openml_dependency_guard() -> None:
    """Ensure tests remain independent of OpenML/network availability."""
    tests_dir = Path(__file__).parent
    forbidden = "fetch_" + "openml("
    offenders: list[str] = []

    for path in tests_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            offenders.append(str(path.relative_to(tests_dir.parent)))

    assert offenders == [], f"OpenML usage found in tests: {offenders}"
