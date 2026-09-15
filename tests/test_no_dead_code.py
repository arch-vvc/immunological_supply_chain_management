"""No function may carry statements after an unconditional return/raise at
its top level (the Stage 27 figure block was silently orphaned that way), and
no `if __name__ == "__main__"` guard may sit inside a function."""
import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "src", "*.py"))) + [
    os.path.join(ROOT, "app.py"), os.path.join(ROOT, "main.py")]


def _dead_after_return(fn):
    for i, st in enumerate(fn.body):
        if isinstance(st, (ast.Return, ast.Raise)) and i < len(fn.body) - 1:
            return fn.body[i + 1].lineno
    return None


def test_no_statements_after_unconditional_return():
    bad = []
    for path in FILES:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line = _dead_after_return(node)
                if line is not None:
                    bad.append(f"{os.path.relpath(path, ROOT)}:{line} in {node.name}()")
    assert not bad, "unreachable code after return:\n  " + "\n  ".join(bad)


def test_main_guard_is_at_module_level():
    bad = []
    for path in FILES:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.If) and "__main__" in ast.dump(sub.test):
                        bad.append(f"{os.path.relpath(path, ROOT)}:{sub.lineno} inside {node.name}()")
    assert not bad, "__main__ guard nested in a function:\n  " + "\n  ".join(bad)
