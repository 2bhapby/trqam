import json
import traceback
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


def main():
    notebook_path = Path("TRQAM_Report.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    shell = InteractiveShell.instance()
    ns = shell.user_ns
    ns["__name__"] = "__main__"

    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[notebook] running code cell {idx}")
        result = shell.run_cell(source, store_history=False)
        if result.error_in_exec is not None:
            print(f"[notebook] failed at cell {idx}")
            traceback.print_exception(
                type(result.error_in_exec),
                result.error_in_exec,
                result.error_in_exec.__traceback__,
            )
            raise SystemExit(1)

    report_path = ns.get("report_path")
    if report_path is None:
        raise SystemExit("[notebook] report_path was not created")
    report_path = Path(report_path)
    if not report_path.exists():
        raise SystemExit(f"[notebook] report file does not exist: {report_path}")

    print(f"[notebook] report ok: {report_path}")


if __name__ == "__main__":
    main()
