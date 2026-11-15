"""The 3D CNN architecture and the cropped-input helper.

Extracted from the original per-experiment training scripts. The docstrings here are
new; the layer definitions and forward passes parse to exactly the tree the originals
did, so the behaviour is unchanged.
"""
import numpy as np
import torch

from scipy.ndimage import find_objects


class Custom3DCNN(torch.nn.Module):
    """The 3D CNN used for every image-only result in the paper.

    Three convolution blocks, each ``Conv3d(3x3x3, padding=1) -> norm -> ReLU ->
    MaxPool3d(2)``, widening the channels ``in_channels -> 16 -> 32 -> 64``. The feature
    map is then collapsed by ``AdaptiveAvgPool3d(1)`` and a single ``Linear`` maps the 64
    channels to the output.

    Deliberately small: with a few thousand metastases and one label each, a deeper
    backbone overfits long before it generalises. The learning-curve sweep
    (:mod:`training.learning_curve`) measures whether the task is limited by data.

    ``num_classes=1`` emits a single logit for the binary task (progression vs not);
    ``num_classes=3`` emits one logit per RANO level. The three pooling stages divide
    each spatial dimension by 8, which is the padding multiple
    :func:`use_cropped_images_fn` has to respect.

    Args:
        num_classes: 1 for binary (a single logit), 3 for the RANO three-class task.
        in_channels: one channel per input sequence, so 1 for T1-post alone and 3 for
            T1-post + T2 + FLAIR.
        norm_type: ``'instance'`` (default), ``'batch'``, ``'group'`` or ``'layer'``.
            Instance norm is the default because the effective batch size is small.
        use_norm: set False to skip normalisation entirely.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        norm_type = 'instance',
        use_norm = True,
    ) -> None:
        """
        Constructor for the Custom3DCNN class.

        Args:
            num_classes (int): Number of output classes.
        """
        super(Custom3DCNN, self).__init__()
        self.num_classes = num_classes
        self.use_norm = use_norm
        filters = [16, 32, 64]
        # 3D convolutional layers
        self.conv1 = torch.nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv3d(filters[0], filters[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv3d(filters[1], filters[2], kernel_size=3, padding=1)
        # Add norm
        if norm_type == 'batch':
            self.bn1 = torch.nn.BatchNorm3d(filters[0])
            self.bn2 = torch.nn.BatchNorm3d(filters[1])
            self.bn3 = torch.nn.BatchNorm3d(filters[2])
        elif norm_type == 'group':
            self.bn1 = torch.nn.GroupNorm(4, filters[0])
            self.bn2 = torch.nn.GroupNorm(8, filters[1])
            self.bn3 = torch.nn.GroupNorm(8, filters[2])
        elif norm_type == 'layer':
            self.bn1 = torch.nn.GroupNorm(1, filters[0])
            self.bn2 = torch.nn.GroupNorm(1, filters[1])
            self.bn3 = torch.nn.GroupNorm(1, filters[2])
        else:
            self.bn1 = torch.nn.InstanceNorm3d(filters[0])
            self.bn2 = torch.nn.InstanceNorm3d(filters[1])
            self.bn3 = torch.nn.InstanceNorm3d(filters[2])
        # Max-pooling layer
        self.pool = torch.nn.MaxPool3d(2)
        # Fully connected layers
        self.global_pooling = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Linear(filters[2], num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model, applying convolutions, max-pooling, and fully connected layers
        """
        x = self.conv1(x)
        if self.use_norm == True:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        if self.use_norm == True:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        if self.use_norm == True:
            x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def use_cropped_images_fn(inputs, batch_data, model, device='cuda', use_cropped_images=True, downsampling_factor=2**3):
    """Run the model on each sample cropped to its own non-zero bounding box.

    A metastasis occupies a small part of its padded volume, so most voxels in a batch
    are zeros. Cropping each sample to its own bounding box removes that waste, but it
    also gives every sample a different shape — which is why samples go through the model
    one at a time and the outputs are concatenated, rather than batched.

    Each cropped volume is padded up to a multiple of `downsampling_factor` (8, matching
    the model's three ``MaxPool3d(2)`` stages) so pooling never hits an odd dimension.
    The bounding box covers the spatial dimensions only — the union of non-zero voxels
    across channels — so a sample with an all-zero channel (a metastasis with no
    T2/FLAIR coverage) keeps the channel count the model expects instead of collapsing
    it.

    With `use_cropped_images=False` the batch passes through unchanged.

    Args:
        inputs: batched volumes, ``(B, C, H, W, D)``.
        batch_data: the batch dict, read only to identify a sample in the warning
            printed when it has no non-zero voxels at all.
        model: the network to apply.
        device: where to place each cropped sample.
        use_cropped_images: False disables cropping.
        downsampling_factor: the multiple each spatial dimension is padded up to.

    Returns:
        The concatenated model outputs, one row per input sample.
    """
    if use_cropped_images == False:
        outputs = model(inputs)
    else:
        inputs_cpu = inputs.detach().cpu().numpy()
        for batch_index in range(0, inputs_cpu.shape[0]):
            temp_batch = inputs_cpu[batch_index, ...]
            volume_location = find_objects((temp_batch != 0).any(axis=0))
            if len(volume_location) > 0:
                temp_batch = temp_batch[(slice(None),) + volume_location[0]]
                temp_batch = np.pad(temp_batch, [(0,0)] + [(int(np.floor((downsampling_factor - (x % downsampling_factor)) / 2)), int(np.ceil((downsampling_factor - (x % downsampling_factor)) / 2))) for x in temp_batch.shape[1:]])
            else:
                print(batch_data['patient_id'][batch_index])
                print(batch_data['met'][batch_index])
            temp_batch = np.expand_dims(temp_batch, axis=0)
            temp_batch = torch.tensor(temp_batch).to(device)
            temp_outputs = model(temp_batch)
            if batch_index == 0:
                outputs = temp_outputs
            else:
                outputs = torch.concatenate([outputs, temp_outputs])
    return outputs
