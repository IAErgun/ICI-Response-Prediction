"""Runtime compatibility shim, applied on import.

PyTorch 2.6 changed ``torch.load`` to default ``weights_only=True``, which cannot
unpickle the MONAI ``MetaTensor`` objects held in the ``PersistentDataset`` cache. The
cache and the checkpoints are files this pipeline wrote itself, so the pre-2.6
behaviour is restored.

Importing this module patches ``torch.load`` — the original training scripts did the
same at their own module level, so the behaviour is unchanged. Any module that loads a
cached dataset or a checkpoint should import it explicitly:

    from . import compat  # noqa: F401 - patches torch.load on import
"""
import torch

_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full
