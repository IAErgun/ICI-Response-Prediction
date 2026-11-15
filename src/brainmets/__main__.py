"""The ``brainmets`` command: one entry point for every subpackage.

    brainmets cohort <command> [options]        e.g. brainmets cohort count-patients
    brainmets training <command> [options]      e.g. brainmets training run t1post-binary
    brainmets evaluation <tool> [options]       e.g. brainmets evaluation bootstrap-test-ci

Each is equivalent to the corresponding ``python -m brainmets.<...>`` invocation, which
keeps working. Tool names accept dashes or underscores. Subpackages are imported only
when used, so ``brainmets cohort`` does not load torch.
"""
import ast
import importlib.util
import pathlib
import runpy
import sys

GROUPS = {
    "cohort": "cohort definition, QC checks, Table 1 aggregates, fold generation",
    "training": "train the 3D CNN experiments",
    "evaluation": "score run outputs: aggregation, test metrics, bootstrap CIs, plots",
}
TOOL_GROUPS = ("evaluation",)         # groups whose commands are one runnable module each
NOT_TOOLS = set()                     # helper modules in those groups, hidden from listings


def _tools(group: str) -> dict[str, str]:
    """The runnable modules of `group`, with the first line of each docstring.

    Docstrings are read from source rather than by importing, so listing the evaluation
    tools does not pull in torch.
    """
    spec = importlib.util.find_spec(f"brainmets.{group}")
    folder = pathlib.Path(spec.submodule_search_locations[0])
    tools = {}
    for path in sorted(folder.glob("*.py")):
        name = path.stem
        if name.startswith("_") or name in NOT_TOOLS:
            continue
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        tools[name.replace("_", "-")] = _summary(doc)
    return tools


def _summary(doc: str, width: int = 64) -> str:
    """The first sentence of a docstring, which may wrap over several lines."""
    first_paragraph = " ".join(doc.strip().split("\n\n")[0].split())
    sentence = first_paragraph.split(". ")[0].rstrip(".")
    return sentence if len(sentence) <= width else sentence[:width - 1].rstrip() + "…"


def _usage(out=None) -> None:
    out = out or sys.stdout                 # looked up per call, so redirection works
    print("usage: brainmets <group> <command> [options]\n", file=out)
    for group, what in GROUPS.items():
        print(f"  {group:12s} {what}", file=out)
    print("\n`brainmets <group> --help` lists a group's commands.", file=out)


def _run_tool(group: str, argv: list[str]) -> int:
    tools = _tools(group)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: brainmets {group} <tool> [options]\n")
        for name, doc in tools.items():
            print(f"  {name:34s} {doc}")
        return 0
    tool = argv[0].replace("_", "-")
    if tool not in tools:
        print(f"brainmets {group}: unknown tool {argv[0]!r}; "
              f"choose from: {', '.join(tools)}", file=sys.stderr)
        return 2
    module = f"brainmets.{group}.{tool.replace('-', '_')}"
    # argparse names the program after sys.argv[0]; alter_sys would overwrite it with
    # the module's file path, so it is left off and argv is set here instead.
    sys.argv = [f"brainmets {group} {tool}", *argv[1:]]
    runpy.run_module(module, run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return 0
    group, rest = argv[0], argv[1:]
    if group == "cohort":
        from .cohort.__main__ import main as cohort_main
        return cohort_main(rest) or 0
    if group == "training":
        from .training.__main__ import main as training_main
        return training_main(rest) or 0
    if group in TOOL_GROUPS:
        return _run_tool(group, rest)
    print(f"brainmets: unknown group {group!r}\n", file=sys.stderr)
    _usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
