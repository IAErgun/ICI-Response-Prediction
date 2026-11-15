"""Tests for the ``brainmets`` command (``brainmets/__main__.py``).

    python -m unittest tests.test_cli -v
"""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))                    # for `fixtures`

from brainmets.__main__ import main  # noqa: E402
from fixtures import synthetic_cohort  # noqa: E402


def run(argv):
    """Run the command in-process; return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    saved_argv = sys.argv[:]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as e:                  # argparse exits on --help
                code = e.code
    finally:
        sys.argv = saved_argv
    return code, out.getvalue(), err.getvalue()


class TestBrainmetsCommand(unittest.TestCase):
    def test_no_arguments_lists_the_groups(self):
        code, out, _ = run([])
        self.assertEqual(code, 0)
        for group in ("cohort", "training", "evaluation"):
            self.assertIn(group, out)

    def test_unknown_group_fails(self):
        code, _, err = run(["nope"])
        self.assertEqual(code, 2)
        self.assertIn("unknown group", err)

    def test_evaluation_lists_its_tools_without_helpers(self):
        code, out, _ = run(["evaluation", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("bootstrap-test-ci", out)
        self.assertIn("compute-test-metrics-subgroups", out)
        self.assertNotIn("__init__", out)

    def test_unknown_tool_names_the_choices(self):
        code, _, err = run(["evaluation", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("bootstrap-test-ci", err)

    def test_tool_accepts_underscores_and_names_itself(self):
        code, out, _ = run(["evaluation", "bootstrap_sequence_comparison_ci", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage: brainmets evaluation bootstrap-sequence-comparison-ci", out)

    def test_cohort_dispatch_runs_a_check(self):
        with synthetic_cohort():
            code, out, _ = run(["cohort", "count-patients"])
        self.assertEqual(code, 0)
        self.assertIn("patient-level split is clean", out)


if __name__ == "__main__":
    unittest.main()
