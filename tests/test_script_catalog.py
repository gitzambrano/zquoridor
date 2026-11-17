import ast

from tools import script_catalog


def test_script_index_is_current():
    assert script_catalog.run(dict(script_catalog.CONFIG, check=True)) == 0


def test_roles_follow_layout():
    runner = ast.parse("CONFIG = {}\ndef main():\n    pass\n")
    module = ast.parse("def helper():\n    pass\n")
    assert script_catalog.role("tools/x.py", runner) == "runner"
    assert script_catalog.role("tools/x.py", module) == "module"
    assert script_catalog.role("tools/campaigns/x.py", runner) == "campaign"
    assert script_catalog.role("colab/playwright/diagnostics/x.py", runner) == "diagnostic"


def test_summary_uses_first_sentence():
    tree = ast.parse('"""First sentence. Second sentence.\n\nMore text."""\n')
    assert script_catalog.summary(tree) == "First sentence."
    assert script_catalog.summary(ast.parse("x = 1\n")) == "No module docstring."


def test_generated_region_replaces_only_marked_text():
    text = f"Head.\n\n{script_catalog.BEGIN}\nold\n{script_catalog.END}\n\nTail.\n"
    out = script_catalog.updated_text(text, f"{script_catalog.BEGIN}\nnew\n{script_catalog.END}\n")
    assert out.startswith("Head.") and out.rstrip().endswith("Tail.")
    assert "old" not in out and "new" in out
