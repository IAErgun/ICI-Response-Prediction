# This file is adapted from MONAI (https://github.com/Project-MONAI/MONAI): the
# `CropForeground` and `CropForegroundd` transforms in monai/transforms/croppad/.
#
#     Copyright (c) MONAI Consortium
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         http://www.apache.org/licenses/LICENSE-2.0
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
# Changes from the MONAI originals: the crop is a box of fixed `roi_size` centred on the
# foreground's bounding box (`compute_bounding_box`), where MONAI crops to the bounding
# box itself; the `margin`, `allow_smaller` and `k_divisible` options are removed
# accordingly. The lazy-evaluation, padding and inverse logic is MONAI's, unchanged.
# A copy of the Apache License 2.0 is in LICENSES/Apache-2.0.txt.
"""Foreground-centred cropping transforms for MONAI pipelines.

A metastasis is a small object inside a large volume, so the models are fed a fixed-size
box centred on the segmented lesion rather than the whole scan. `ForegroundCenteredCrop`
implements that crop on arrays; `ForegroundCenteredCropd` is the dictionary-style wrapper
the image pipeline uses (see :func:`brainmets.training.data.get_transforms`).

Both are adapted from MONAI's `CropForeground` / `CropForegroundd` (Apache-2.0; see the
notice at the top of this file).
"""
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, Hashable, Mapping, Optional, Union

import numpy as np
import torch
from monai.config import IndexSelection, KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import BorderPad, Crop, Pad
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.transform import LazyTransform
from monai.transforms.utils import generate_spatial_bounding_box, is_positive
from monai.utils import (LazyAttr, PytorchPadMode, TraceKeys,
                         convert_data_type, ensure_tuple, ensure_tuple_rep)


class ForegroundCenteredCrop(Crop):
    """Crop an image using a bounding box with a predetermined size.

    The bounding box is generated from a foreground region of interest by finding the center
    of the bounding box of the ROI, then centering the fixed-size crop region on that point.
    The typical usage is to help training and evaluation if the valid part is small in the
    whole medical image. Users can define arbitrary function to select expected foreground
    from the whole image or specified channels.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling
    topic<lazy_resampling>`for more information.

    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        select_fn: Callable = is_positive,
        channel_indices: IndexSelection | None = None,
        return_coords: bool = False,
        mode: str = PytorchPadMode.CONSTANT,
        lazy: bool = False,
        **pad_kwargs: Any,
    ) -> None:
        """Crop an image using a bounding box with a predetermined size.

        Args:
            roi_size: Fixed size of the output cropped region.
            select_fn: function to select expected foreground, default is
                to select values > 0.
            channel_indices: if defined, select foreground only on the
                specified channelsof image. if None, select foreground on the whole image.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``,
                ``"maximum"``, ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``,
                ``"symmetric"``, ``"wrap"``, ``"empty"``} available modes for PyTorch Tensor:
                {``"constant"``, ``"reflect"``,``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to
                ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False.
            pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        LazyTransform.__init__(self, lazy)
        self.select_fn = select_fn
        self.channel_indices = (
            ensure_tuple(channel_indices) if channel_indices is not None else None
        )
        self.return_coords = return_coords
        self.roi_size = roi_size
        self.padder = Pad(mode=mode, lazy=lazy, **pad_kwargs)

    @Crop.lazy.setter  # type: ignore
    def lazy(self, _val: bool) -> None:
        """Set the lazy property."""
        self._lazy = _val
        self.padder.lazy = _val

    @property
    def requires_current_data(self) -> bool:
        """Whether the transform requires current data."""
        return False

    def compute_bounding_box(self, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Compute the start points and end points of bounding box to crop."""
        box_start, box_end = generate_spatial_bounding_box(
            img,
            self.select_fn,
            self.channel_indices,
            0,
            False,
        )
        roi_size_, *_ = convert_data_type(
            self.roi_size, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True
        )
        box_start_, *_ = convert_data_type(
            box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True
        )
        box_end_, *_ = convert_data_type(
            box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True
        )
        orig_spatial_size = box_end_ - box_start_
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(
            np.asarray(self.roi_size) - orig_spatial_size, 2
        )
        box_end_ = box_start_ + roi_size_
        return box_start_, box_end_

    def crop_pad(
        self,
        img: torch.Tensor,
        box_start: np.ndarray,
        box_end: np.ndarray,
        mode: str | None = None,
        lazy: bool = False,
        **pad_kwargs: Any,
    ) -> torch.Tensor:
        """Crop and pad based on the bounding box."""
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices, lazy=lazy)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(
            box_end
            - np.asarray(
                img.peek_pending_shape()
                if isinstance(img, MetaTensor)
                else img.shape[1:]
            ),
            0,
        )
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(
            cropped.peek_pending_shape()
            if isinstance(cropped, MetaTensor)
            else cropped.shape[1:]
        )
        ret = self.padder.__call__(
            img=cropped, to_pad=pad_width, mode=mode, lazy=lazy, **pad_kwargs
        )
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta() and isinstance(ret, MetaTensor):
            if not lazy:
                ret.applied_operations[-1][TraceKeys.EXTRA_INFO][
                    "pad_info"
                ] = ret.applied_operations.pop()
            else:
                pad_info = ret.pending_operations.pop()
                crop_info = ret.pending_operations.pop()
                extra = crop_info[TraceKeys.EXTRA_INFO]
                extra["pad_info"] = pad_info
                self.push_transform(
                    ret,
                    orig_size=crop_info.get(TraceKeys.ORIG_SIZE),
                    sp_size=pad_info[LazyAttr.SHAPE],
                    affine=crop_info[LazyAttr.AFFINE] @ pad_info[LazyAttr.AFFINE],
                    lazy=lazy,
                    extra_info=extra,
                )
        return ret

    def __call__(  # type: ignore[override]
        self,
        img: torch.Tensor,
        mode: str | None = None,
        lazy: bool | None = None,
        **pad_kwargs: Any,
    ) -> torch.Tensor:

        box_start, box_end = self.compute_bounding_box(img)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = self.crop_pad(img, box_start, box_end, mode, lazy=lazy_, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end  # type: ignore[return-value]
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        """Apply the inverse of the transform, uncropping the image."""
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)


class ForegroundCenteredCropd(Cropd):
    """Dictionary-based version :py:class:`transforms.ForegroundCenteredCropd`.

    Crop a fixed-size region, centered on the center of the foreground.
    The typical usage is to help training and evaluation if the valid part is small
    in the whole medical image. The foreground part can be determined by any field in the
    data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all
        `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source
    image or specified channels.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling
    topic<lazy_resampling>` for more information.
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        roi_size: Union[Sequence[int], int],
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        mode: Union[Sequence[str], str] = PytorchPadMode.CONSTANT,
        start_coord_key: Optional[str] = "foreground_start_coord",
        end_coord_key: Optional[str] = "foreground_end_coord",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **pad_kwargs: Any,
    ) -> None:
        """Dictionary-based version :py:class:`transforms.ForegroundCenteredCropd`.

        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be
                image or label, etc.
            roi_size: Size of the fixed size region to crop.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``,
                ``"maximum"``, ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``,
                ``"symmetric"``, ``"wrap"``, ``"empty"``} available modes for PyTorch Tensor:
                {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to
                ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            start_coord_key: key to record the start coordinate of spatial bounding box foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False.
            pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = ForegroundCenteredCrop(
            select_fn=select_fn,
            channel_indices=channel_indices,
            roi_size=roi_size,
            lazy=lazy,
            **pad_kwargs,
        )
        super().__init__(
            keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        """Set the lazy property of the transform."""
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self) -> bool:
        """Whether the transform requires current data."""
        return True

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: Optional[bool] = None
    ) -> dict[Hashable, torch.Tensor]:
        """Apply the transform."""
        d = dict(data)
        self.cropper: ForegroundCenteredCrop
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(
                img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_
            )
        return d


