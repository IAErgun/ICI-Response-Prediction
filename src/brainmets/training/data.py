"""Data list construction, MONAI transform pipelines and class-balancing sampler.

Extracted from the original per-experiment training scripts with an identical parse
tree; the behaviour is unchanged.
"""
import os
import torch

from .crop import ForegroundCenteredCropd
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from monai.transforms import (Compose, DeleteItemsd, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, RandFlipd, ToDeviced)
from .transforms import MapROIValues, MaskedOffsetTransform


def load_data(df, base_dir, class_dict, seq_of_interest, response_label_met, roi_filename = "_T1Post_TRACKED-label.nii.gz", t1post_filename = "_T1Post.nii.gz", t2_filename = "_T2.nii.gz", t1pre_filename = "_T1Pre.nii.gz", flair_filename = "_FLAIR.nii.gz", adc_filename = "_ADC.nii.gz", swi_filename = "_SWI.nii.gz", dwi_filename = "_DWI.nii.gz"):
    """
    Load and process data for metastasis detection.

    Args:
        df (DataFrame): Input DataFrame with patient and metastasis information.

    Returns:
        list: A list of dictionaries, each containing information about a metastasis.
    """
    # Create an empty list to store data dictionaries
    data_list = []
    # Work on a copy to avoid SettingWithCopyWarning on the sliced input df
    df = df.copy()
    # Convert the 'Patient_ID' column to strings
    df.Patient_Number = df.Patient_Number.astype(str)
    # Get unique Patient_IDs from the DataFrame
    patient_ids = df.Patient_Number.unique().tolist()
    # Iterate through patient directories
    for patient_id in tqdm(patient_ids, desc="load_data patients"):
        # Get subset of df for the current patient
        patient_df = df.loc[df.Patient_Number == str(patient_id)]
        for met_number in list(patient_df.Metastasis_ID):
            met_df = patient_df.loc[patient_df.Metastasis_ID == met_number]
            met_df_file_path = os.path.split(met_df.PreTreatment_Image.values[0])[0]
            split_path = met_df_file_path.split('/')
            sequence_map = {
                "t1post": t1post_filename,
                "t1pre": t1pre_filename,
                "flair": flair_filename,
                "t2": t2_filename,
                "swi": swi_filename,
                "dwi": dwi_filename,
                "adc": adc_filename,
            }
            patient_files = sorted(os.listdir(met_df_file_path))
            seq_paths = {}
            for seq, ext in sequence_map.items():
                for file_counter, file in enumerate(patient_files):
                    if (ext == file) and (seq in seq_of_interest):
                        seq_paths[f"{seq}_path"] = os.path.join(met_df_file_path, file)
                        break
            gt_label = met_df[response_label_met].values[0]
            gt_label = class_dict[gt_label]
            # Create a dictionary with the path to the t1post image, path to the ROI, metastasis_id, and label
            data_dict = {
                "roi": os.path.join(met_df_file_path, roi_filename),
                "met": met_number,
                "treatment_round": split_path[-2],
                "label": gt_label,
                "patient_id": split_path[-3],
            }
            for seq, path in [(k, v) for k, v in sequence_map.items() if k in seq_of_interest]:
                data_dict[seq] = seq_paths[f"{seq}_path"]
            data_list.append(data_dict)
    return data_list


def get_transforms(bb_size, seq_of_interest, mode="train", use_gpu=True):
    """
    Get data transformation pipeline.

    Args:
        bb_size (int): Bounding box size.
        mode (str): "train" or "validation" mode.

    Returns:
        monai.transforms.Compose: A composition of data transforms.
    """
    # Idea: First use deterministic transform -> cache to disk -> then use random transform
    # Constraint: RandBiasField needs to be applied on entire image
    # Move to GPU before first random transform
    # Then use lazy transforms before using non-lazy transforms
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    bb_size_padded = int(bb_size * 1.1)
    if mode == "train":
        initial_crop_size = (bb_size_padded, bb_size_padded, bb_size_padded)
    else:
        initial_crop_size = (bb_size, bb_size, bb_size)
    transform = [
        LoadImaged(keys=seq_of_interest + ["roi"]),
        ToDeviced(device=device, keys=seq_of_interest + ["roi"]),
        EnsureChannelFirstd(keys=seq_of_interest + ["roi"]),
        NormalizeIntensityd(keys=seq_of_interest),
        MapROIValues(roi_key="roi", dict_key="met"),
        MaskedOffsetTransform(roi_key="roi", image_keys=seq_of_interest, offset=5),
        ForegroundCenteredCropd(
            keys=seq_of_interest + ["roi"],
            source_key="roi",
            roi_size=initial_crop_size,
            lazy=False,
            ),
        DeleteItemsd(keys=["roi"]),
    ]
    if mode == "train": # change from "train" to False for debugging script speed without transforms
        transform.extend(
            [
                # Augmentation is flips only. Sequence dropout, random bias field,
                # random spatial crop, rotation, contrast adjustment, Gaussian noise and
                # Gaussian smoothing were all tried during development and none improved
                # validation performance, so they are not part of the pipeline.
                RandFlipd(keys=seq_of_interest, spatial_axis=[0], prob=0.5, lazy=False),
                RandFlipd(keys=seq_of_interest, spatial_axis=[1], prob=0.5, lazy=False),
                RandFlipd(keys=seq_of_interest, spatial_axis=[2], prob=0.5, lazy=False),
            ]
        )
    return Compose(transform, lazy=None)


def define_sampler(train_data_list, weighting=None):
    train_labels = [x["label"] for x in train_data_list]
    train_label_type = list(set(train_labels))
    train_class_count = {str(_k): train_labels.count(_k) for _k in train_label_type}
    if weighting == None:
        weights = {_k: 1.0 / train_class_count[_k] for _k in train_class_count.keys()}
    else:
        weights = {"0": 1.0 / (train_class_count["0"] + weighting), "1": 1.0 / (train_class_count["1"] - weighting)}
    samples_weights = torch.tensor(
        [weights[str(x)] for x in train_labels], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
    )
    return sampler
