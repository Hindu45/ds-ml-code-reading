"""Install only packages missing from the base Python distribution.

Workflow
--------
1. Venv guard          — if not already running inside the project venv,
                         locate it (by ``pyvenv.cfg`` marker) and re-exec
                         this script under it.
2. Site-packages gate  — ensure ``pyvenv.cfg`` has
                         ``include-system-site-packages = true``; if the
                         flag is absent or false the file is patched and the
                         script re-execs so the updated config takes effect
                         (Python reads ``pyvenv.cfg`` at startup only).
3. Read requirements   — parse the requirements file, ignoring blank lines,
                         comments, ``-r`` includes, ``--index-url`` flags,
                         URL-based specs, and environment markers.
4. Diff against installed — compare each requirement name (PEP 503
                         normalised: ``[-_.]`` → ``-``, lowercased) against
                         the packages visible to the current interpreter.
5. Install gap         — run ``pip install`` for every missing package, or
                         just print the list when ``--dry-run`` is given.

Usage:
    python install-missing.py [--dry-run] [--requirements FILE]
"""
import argparse
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path


def _normalize(name: str) -> str:
    """PEP 503: collapse [-_.] runs to '-' and lowercase."""
    return re.sub(r"[-_.]+", "-", name).lower()


def get_installed_packages() -> set[str]:
    """Return PEP 503-normalized names of all packages visible to this interpreter."""
    return {_normalize(dist.metadata["Name"]) for dist in distributions()}


def parse_requirement(line: str) -> str | None:
    """Return normalized package name from a requirements.txt line, or None to skip."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Skip -r includes, --index-url flags, and URL requirements
    if line.startswith("-") or "://" in line:
        return None
    line = line.split("#")[0].strip()   # strip inline comment
    line = line.split(";")[0].strip()   # strip environment markers
    name = re.split(r"[\[=<>!~@\s]", line)[0]
    return _normalize(name) if name else None


def _find_project_venv_python() -> Path | None:
    """Find a venv Python anywhere inside the project directory.

    Uses pyvenv.cfg presence as the canonical marker — works regardless of
    what the venv directory is named (.venv, venv, env, my-env, etc.).
    """
    root = Path(__file__).parent
    for cfg in sorted(root.glob("*/pyvenv.cfg")):  # sorted for determinism
        for candidate in [
            cfg.parent / "Scripts" / "python.exe",  # Windows
            cfg.parent / "bin" / "python",           # Unix
        ]:
            if candidate.exists():
                return candidate
    return None


def _in_project_venv() -> bool:
    """Return True if the current interpreter is a venv rooted inside the project."""
    if sys.prefix == sys.base_prefix:
        return False  # not in any venv
    project_root = Path(__file__).parent.resolve()
    return Path(sys.prefix).resolve().is_relative_to(project_root)


def _ensure_venv() -> None:
    """Re-exec this script under the project venv if not already running there.

    Prevents accidental installation into system Python or a different venv,
    regardless of how the user invoked the script.
    """
    if _in_project_venv():
        return
    venv_py = _find_project_venv_python()
    if venv_py is None:
        sys.exit(
            "Error: no venv in local project found.\n"
            "Create one first to use this script."
        )
    print(f"[install-missing] Re-launching under project venv: {venv_py}")
    raise SystemExit(
        subprocess.run(
            [str(venv_py), __file__, *sys.argv[1:]], check=False
        ).returncode
    )


def _ensure_system_site_packages() -> None:
    """Ensure the active venv's pyvenv.cfg has include-system-site-packages = true.

    Uses sys.prefix to locate the cfg — works for any venv name.
    If the cfg is changed, re-execs the script so the new process starts
    with the updated config already in effect (Python reads pyvenv.cfg at
    startup only, so a warning-and-continue approach would leave the current
    process blind to system packages).
    """
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if not cfg.exists():
        return

    text = cfg.read_text(encoding="utf-8")
    if "include-system-site-packages = true" in text:
        return

    if "include-system-site-packages" in text:
        updated = re.sub(
            r"include-system-site-packages\s*=\s*\S+",
            "include-system-site-packages = true",
            text,
        )
    else:
        updated = text.rstrip("\n") + "\ninclude-system-site-packages = true\n"

    cfg.write_text(updated, encoding="utf-8")
    print("[pyvenv.cfg] Enabled include-system-site-packages — re-launching to pick up the change.", flush=True)
    raise SystemExit(
        subprocess.run([sys.executable, __file__, *sys.argv[1:]], check=False).returncode
    )


def main() -> int:
    _ensure_system_site_packages()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be installed without actually installing",
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        metavar="FILE",
        help="Requirements file to read (default: requirements.txt)",
    )
    args = parser.parse_args()

    req_file = Path(args.requirements)
    if not req_file.exists():
        print(f"Error: {req_file} not found")
        return 1

    installed = get_installed_packages()
    to_install: list[str] = []
    skipped: list[str] = []

    with req_file.open() as f:
        for line in f:
            pkg_name = parse_requirement(line)
            if pkg_name:
                if pkg_name in installed:
                    skipped.append(pkg_name)
                else:
                    to_install.append(line.split("#")[0].strip())

    if skipped:
        print(f"Skipping {len(skipped)} already-installed packages:")
        for pkg in sorted(skipped)[:10]:
            print(f"  - {pkg}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if not to_install:
        print("\nAll packages already installed!")
        return 0

    if args.dry_run:
        print(f"\n[dry-run] Would install {len(to_install)} missing package(s):")
        for pkg in to_install:
            print(f"  + {pkg}")
        return 0

    print(f"\nInstalling {len(to_install)} missing package(s)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *to_install],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    _ensure_venv()
    sys.exit(main())
