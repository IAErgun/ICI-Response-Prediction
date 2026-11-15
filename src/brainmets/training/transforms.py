"""Custom MONAI transforms for the metastasis image pipeline.

The dataset is one row per metastasis inside a volume that may contain several, so the
pipeline needs two operations MONAI does not ship:

- `MapROIValues` reduces a multi-label segmentation to the single metastasis a row refers
  to, which is what makes per-lesion labels usable.
- `MaskedOffsetTransform` zeroes voxels outside a dilated mask around that metastasis, so
  a neighbouring lesion cannot leak into the input.

Both are used by :func:`brainmets.training.data.get_transforms`.
"""
from typing import Dict

import monai
import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure


class MapROIValues(monai.transforms.Transform):
    def __init__(self, roi_key: str, dict_key: str) -> None:
        """
        Constructor for the MapROIValues class.

        Args:
            roi_key (str): Key to access the Region of Interest (ROI) data in the input data dictionary.
            dict_key (str): Key to access the dictionary value to compare with the ROI data.
        """
        self.roi_key = roi_key  # Store the provided ROI key
        self.dict_key = dict_key  # Store the provided dictionary key

    def __call__(self, data: Dict) -> Dict:
        """
        Apply the transform on the input data.

        Args:
            data (dict): Input data dictionary containing various keys and associated values.

        Returns:
            dict: The modified input data dictionary with ROI values updated.
        """
        roi = data[self.roi_key]  # Extract the ROI data from the input data
        met = data.get(
            self.dict_key
        )  # Extract the dictionary value to compare with ROI

        if met is not None:
            met = int(met)  # Convert the dictionary value to an integer
            roi = (
                roi == met
            ).float()  # Set values to 1 where they match with the dictionary value, 0 otherwise

        data[self.roi_key] = roi  # Update the ROI data in the input data dictionary
        return data  # Return the modified input data dictionary


class MaskedOffsetTransform(monai.transforms.Transform):
    """
    This transform modifies the images in a dictionary by zeroing out pixels that are not covered by the
    'roi' mask and an additional surrounding area defined by 'offset'.

    Parameters:
    - data: dict, containing multiple image keys.
    - offset: int, number of voxels to extend around the segmentation mask.
    """

    def __init__(self, roi_key: str, image_keys: list, offset=0):
        self.roi_key = roi_key
        self.image_keys = image_keys
        self.offset = offset

    def __call__(self, data: Dict):
        for image_key in self.image_keys:
            device = data[image_key].device
            image = data[image_key].cpu().numpy()
            mask = data[self.roi_key].cpu().numpy()

            # Create a structuring element with the appropriate size for dilation
            struct_elem = generate_binary_structure(len(mask.shape), 1)
            # Dilate the mask by the specified offset
            dilated_mask = binary_dilation(
                mask, structure=struct_elem, iterations=self.offset
            )
            image_size = np.prod(image.shape)
            dilated_mask_size = np.prod(dilated_mask.shape)

            # If sizes do not match
            if image_size != dilated_mask_size:
                # If dilated_mask is smaller, pad it
                if dilated_mask_size < image_size:
                    crop_size = np.subtract(image.shape, dilated_mask.shape) / 2
                    image = image[
                        :,
                        int(np.ceil(crop_size[1])) : int(np.ceil(-crop_size[1])),
                        int(np.ceil(crop_size[2])) : int(np.ceil(-crop_size[2])),
                        int(np.ceil(crop_size[3])) : int(np.ceil(-crop_size[3])),
                    ]

                elif dilated_mask_size > image_size:
                    crop_size = np.subtract(dilated_mask.shape, image.shape) / 2
                    dilated_mask = dilated_mask[
                        :,
                        int(np.ceil(crop_size[1])) : int(np.ceil(-crop_size[1])),
                        int(np.ceil(crop_size[2])) : int(np.ceil(-crop_size[2])),
                        int(np.ceil(crop_size[3])) : int(np.ceil(-crop_size[3])),
                    ]
            try:
                # Apply the dilated mask to the image
                modified_image = image * dilated_mask
                # Update the image in the dictionary
                data[image_key] = torch.tensor(modified_image, device=device)
            except Exception:
                modified_image = mask
                modified_image[modified_image != 0] = 0
                data[image_key] = torch.tensor(modified_image, device=device)
        return data


