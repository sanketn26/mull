"""Keeps every function and method in the package, examples and tests at 50 lines or fewer."""
import ast
import pathlib
import unittest

MAX_LINES = 50
ROOT = pathlib.Path(__file__).resolve().parents[1]
# Vendored third-party code is kept as shipped upstream.
EXCLUDE = {ROOT / "mull" / "tokenization_baichuan.py"}


def long_functions(limit=MAX_LINES):
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        if path in EXCLUDE or ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                length = node.end_lineno - start + 1
                if length > limit:
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name} ({length} lines)")
    return found


class TestFunctionLength(unittest.TestCase):
    def test_no_function_exceeds_the_limit(self):
        self.assertEqual(long_functions(), [], f"split these into helpers (limit {MAX_LINES} lines)")


if __name__ == "__main__":
    unittest.main()
