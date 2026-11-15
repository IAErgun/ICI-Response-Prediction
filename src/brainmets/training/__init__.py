"""3D CNN training for brain-metastasis ICI response prediction.

One experiment is one :class:`~brainmets.training.config.RunConfig`: the image-only runs
(T1-post, multisequence and T2, each binary or 3-class) and the learning-curve sweep are
presets over shared modules.

Modules
-------
`config`
    `RunConfig` and the experiment presets.
`models`
    The 3D CNN and the cropped-input helper.
`data`
    Data list construction, the MONAI transform pipeline, class-balancing sampler.
`transforms`, `crop`
    The custom MONAI transforms that pipeline uses: per-lesion label selection, masking
    of neighbouring lesions, and the lesion-centred crop.
`engine`
    The training and evaluation loops.
`cohort_input`
    Cohort table -> per-fold data lists, over the shared `cohort` exclusion rules.
`runner`
    The 5-fold cross-validation run.
`learning_curve`
    Validation metric as a function of training-set size.
`compat`
    Restores pre-2.6 `torch.load` behaviour so the MONAI dataset cache unpickles.

`models`, `data` and `engine` were extracted from the original scripts verbatim, so
the modelling behaviour is unchanged by the restructuring.

    python -m brainmets.training presets
    python -m brainmets.training run t1post-binary
"""
from . import config
from .config import PRESETS, RunConfig

__all__ = ["config", "PRESETS", "RunConfig"]
