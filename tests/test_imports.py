"""Every module in the package imports, and importing one has no side effects.

The fast CI job installs without the deep-learning stack, so on its own it never loads
the trainer, the data pipeline or the prediction exporters: a broken import there would
pass. This test imports everything it can. Modules whose optional dependency is missing
are skipped one by one, so the same file is useful with or without torch; CI's
full-stack job runs it with everything installed and skips nothing.

The imports happen in a fresh interpreter started in an empty directory. That keeps the
check honest (nothing is already cached from another test) and catches a module that
writes files when imported.

    python -m unittest tests.test_imports -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

# Third-party packages a module may legitimately need that the fast CI job leaves out.
OPTIONAL = ["torch", "monai", "tensorboard", "nibabel", "tqdm", "scipy"]

PROBE = """
import importlib, json, os, pkgutil, sys
import brainmets
optional = set(json.loads(sys.argv[1]))
report = {}
for m in pkgutil.walk_packages(brainmets.__path__, "brainmets."):
    if m.name.endswith("__main__"):
        continue
    try:
        importlib.import_module(m.name)
        report[m.name] = "ok"
    except ModuleNotFoundError as e:
        missing = (e.name or "").split(".")[0]
        report[m.name] = f"skip:{missing}" if missing in optional else f"fail:{e!r}"
    except BaseException as e:                       # SystemExit included
        report[m.name] = f"fail:{e!r}"
print(json.dumps({"modules": report, "written": sorted(os.listdir("."))}))
"""


class TestEveryModuleImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, MPLBACKEND="Agg")
            # no input or output location is configured: importing must not need one
            for var in [v for v in env if v.startswith("BM_")]:
                del env[var]
            done = subprocess.run([sys.executable, "-c", PROBE, json.dumps(OPTIONAL)],
                                  cwd=empty, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise AssertionError(f"the import probe itself failed:\n{done.stderr}")
        cls.result = json.loads(done.stdout.strip().splitlines()[-1])

    def test_the_whole_package_was_walked(self):
        for expected in ("brainmets.cohort.data", "brainmets.training.engine",
                         "brainmets.evaluation.bootstrap_test_ci"):
            self.assertIn(expected, self.result["modules"])

    def test_each_module_imports(self):
        for name, outcome in sorted(self.result["modules"].items()):
            with self.subTest(module=name):
                if outcome.startswith("skip:"):
                    self.skipTest(f"{outcome[5:]} is not installed")
                self.assertEqual(outcome, "ok")

    def test_importing_writes_nothing(self):
        self.assertEqual(self.result["written"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
