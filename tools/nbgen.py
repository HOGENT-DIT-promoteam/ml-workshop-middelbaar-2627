"""Hulpfuncties om een leerlingen- en oplossingsversie van een notebook te genereren.

Codecellen kunnen oplossingsblokken bevatten:

    # >>> TODO: hint voor de leerling
    oplossing = 42
    # ---
    oplossing = ...          (optioneel: wat de leerling te zien krijgt)
    # <<<
"""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _student(src):
    out, mode, hint = [], None, None
    for line in src.splitlines():
        s = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if s.startswith("# >>>"):
            mode, hint = "sol", s[5:].strip()
            out.append(f"{indent}# {hint}")
        elif s == "# ---" and mode == "sol":
            mode = "stud"
        elif s == "# <<<":
            if mode == "sol":  # geen leerlingversie opgegeven
                out.append(f"{indent}...")
            mode = None
        elif mode == "sol":
            continue
        else:
            out.append(line)
    return "\n".join(out)


def _solution(src):
    out, mode = [], None
    for line in src.splitlines():
        s = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if s.startswith("# >>>"):
            mode = "sol"
            out.append(f"{indent}# {s[5:].strip()}")
        elif s == "# ---" and mode == "sol":
            mode = "stud"
        elif s == "# <<<":
            mode = None
        elif mode == "stud":
            continue
        else:
            out.append(line)
    return "\n".join(out)


def build(cells, path_student, path_solution):
    for path, conv in [(path_student, _student), (path_solution, _solution)]:
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        }
        for kind, src in cells:
            src = src.strip("\n")
            if kind == "md":
                nb.cells.append(new_markdown_cell(src))
            else:
                nb.cells.append(new_code_cell(conv(src)))
        nbformat.write(nb, path)
