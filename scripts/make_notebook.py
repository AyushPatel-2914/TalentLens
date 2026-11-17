"""Convert a '# %%'-style Python file into a Jupyter notebook.

Usage:
    python scripts/make_notebook.py notebooks/finetune_reranker.py

The .py file is the source of truth (easy to read and diff); the .ipynb is generated for Colab.
Lines like '# !pip install ...' become '!pip install ...' in the notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)})
        else:
            body = "\n".join(l[2:] if l.startswith("# !") else l for l in body.splitlines())
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(keepends=True)})

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "accelerator": "GPU", "colab": {"gpuType": "T4"}}}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", convert(Path(arg)))
