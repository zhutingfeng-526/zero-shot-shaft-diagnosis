"""Check local dependencies, importability, numeric assets, and required files."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib
import json
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(run_tests=False):
    result = {"scope": "Installation, asset integrity and optional synthetic contract tests; not a full experimental rerun",
              "dependencies": {}, "imports": {}, "assets": {}, "missing": [],
              "experimental_results_verified_by_this_command": False}
    for name in ("numpy", "scipy", "torch"):
        try:
            module = importlib.import_module(name)
            result["dependencies"][name] = getattr(module, "__version__", "available")
        except ImportError as exc:
            result["dependencies"][name] = str(exc)
            result["missing"].append("Python dependency: " + name)
    if "Python dependency: torch" not in result["missing"]:
        try:
            from shaft_diagnosis.runtime import runtime_report
            result["runtime"] = runtime_report()
        except Exception as exc:
            result["runtime"] = {"error": type(exc).__name__ + ": " + str(exc)}
            result["missing"].append("Runtime inspection failed")
    for path in sorted((ROOT / "src" / "shaft_diagnosis").glob("*.py")):
        name = "shaft_diagnosis." + path.stem
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            importlib.import_module(name)
            result["imports"][name] = "ok"
        except Exception as exc:
            result["imports"][name] = type(exc).__name__ + ": " + str(exc)
            result["missing"].append("Module unavailable: " + name)
    manifest_path = ROOT / "release_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            state = "missing"
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                state = "ok" if digest == entry["sha256"] else "checksum mismatch"
            result["assets"][entry["path"]] = state
            if state != "ok":
                result["missing"].append(entry["path"] + ": " + state)
        result["missing"].extend(manifest.get("missing_required_assets", []))
    else:
        result["missing"].append("release_manifest.json")
    if run_tests:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        tests = unittest.TextTestRunner(verbosity=2).run(suite)
        result["tests"] = {"run": tests.testsRun, "failures": len(tests.failures),
                           "errors": len(tests.errors), "skipped": len(tests.skipped),
                           "successful": tests.wasSuccessful()}
    result["ready_for_complete_inference"] = not result["missing"]
    if run_tests and not result["tests"]["successful"]:
        result["ready_for_complete_inference"] = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    output = check(args.tests)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    sys.exit(0 if output["ready_for_complete_inference"] else 2)
