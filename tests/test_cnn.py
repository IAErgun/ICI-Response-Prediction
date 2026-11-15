"""Forward and backward pass checks for the 3D CNN.

The class definition is read out of `brainmets/training/models.py` by AST and executed
in isolation, so the test exercises the real source without triggering that package's
heavy imports (monai and the rest), which the networks themselves do not need.

Needs torch; skipped cleanly if it is not installed, so

    python -m unittest discover -s tests

works whether or not the deep-learning stack is present.
"""
import ast
import importlib.util
import os
import unittest

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                  # pragma: no cover
    HAVE_TORCH = False

NORM_TYPES = ["instance", "batch", "group", "layer"]


def _source_path():
    """Locate models.py through the installed package.

    Found via the package spec rather than a path relative to this file, so the test
    works wherever brainmets is installed from.
    """
    spec = importlib.util.find_spec("brainmets.training")
    if spec is None or not spec.submodule_search_locations:
        raise unittest.SkipTest("brainmets is not importable — run `pip install -e .`")
    return os.path.join(spec.submodule_search_locations[0], "models.py")


def load_cnn_class():
    """Execute just the class definition, with torch as its only dependency."""
    path = _source_path()
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    ns = {"torch": torch}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Custom3DCNN":
            exec(compile(ast.get_source_segment(src, node), f"<{node.name}>", "exec"), ns)
    if "Custom3DCNN" not in ns:
        raise AssertionError("Custom3DCNN not found in models.py")
    return ns["Custom3DCNN"]


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {"Custom3DCNN": load_cnn_class()}

    def test_output_shape_is_batch_by_classes(self):
        """Every norm type and class count must give (batch, num_classes)."""
        for name, cls in self.models.items():
            for norm_type in NORM_TYPES:
                for use_norm in (True, False):
                    for num_classes in (1, 3):
                        with self.subTest(model=name, norm=norm_type,
                                          use_norm=use_norm, classes=num_classes):
                            model = cls(num_classes=num_classes, in_channels=3,
                                        norm_type=norm_type, use_norm=use_norm)
                            model.eval()
                            with torch.no_grad():
                                out = model(torch.randn(2, 3, 32, 32, 32))
                            self.assertEqual(tuple(out.shape), (2, num_classes))

    def test_accepts_one_channel_per_sequence(self):
        """in_channels is the number of input sequences: 1 for T1-post, 3 for multiseq."""
        for name, cls in self.models.items():
            for in_channels in (1, 3):
                with self.subTest(model=name, in_channels=in_channels):
                    model = cls(num_classes=1, in_channels=in_channels)
                    model.eval()
                    with torch.no_grad():
                        out = model(torch.randn(2, in_channels, 32, 32, 32))
                    self.assertEqual(tuple(out.shape), (2, 1))

    def test_backward_populates_gradients(self):
        """A full training step: gradients must reach the first and last layer."""
        for name, cls in self.models.items():
            for norm_type in NORM_TYPES:
                with self.subTest(model=name, norm=norm_type):
                    model = cls(num_classes=2, in_channels=3,
                                norm_type=norm_type, use_norm=True)
                    model.train()
                    out = model(torch.randn(2, 3, 32, 32, 32))
                    loss = torch.nn.functional.cross_entropy(
                        out, torch.tensor([0, 1]))
                    loss.backward()
                    self.assertIsNotNone(model.conv1.weight.grad)
                    self.assertIsNotNone(model.fc1.weight.grad)
                    self.assertTrue(torch.isfinite(loss))

    def test_downsamples_spatially_by_eight(self):
        """Three pooling stages, which is the padding multiple the crop helper uses."""
        for name, cls in self.models.items():
            with self.subTest(model=name):
                model = cls(num_classes=1, in_channels=1)
                model.eval()
                # 8 is the smallest size that survives three halvings
                with torch.no_grad():
                    out = model(torch.randn(1, 1, 8, 8, 8))
                self.assertEqual(tuple(out.shape), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
