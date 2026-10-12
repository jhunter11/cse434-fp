import json
from pathlib import Path
import sys
from types import SimpleNamespace

from captionbench.data import read_jsonl


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "baseline.ipynb"


def code_cells():
    return ["".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def test_notebook_code_compiles_without_saved_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_never_scores_stale_partial_model_results(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace())
    (tmp_path / "generic-test.jsonl").write_text("{}\n", encoding="utf-8")
    called = []
    namespace = {
        "run": tmp_path, "selected": [{"image_id": "one"}], "models": ["generic", "blip-base"],
        "RUN_SPLIT": "test", "predictions": [{"image_id": "one", "model": "generic", "caption": "stale"}],
        "evaluate": lambda *args: called.append(args), "read_jsonl": read_jsonl, "json": json,
    }
    source = next(code for code in code_cells() if "# @title Score the shared test images" in code)
    exec(compile(source, "notebook-score-cell", "exec"), namespace)
    assert called == []
    assert namespace["scores"] is None
    assert namespace["predictions"] == []
