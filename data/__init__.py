from flairsyn.lib.datasets import create_loaders, get_datasets
from typing import Union, Tuple, Optional, Sequence
from monai import transforms
import os
from torch.utils.data import DataLoader


def create_loaders(
    batch_size: int = 24,
    roi_size: int = 128,
    dataset: str = "../data/RS/RS_train_split.json",
    data_dir: str = "../data/RS/conformed",
    img_size: Union[Tuple[int, int], int] = (224, 224),
    slicing_direction: str = "axial",
    slice_thickness: Optional[int] = 1,
    skull_strip: float = 0.0,
    cache: str = "persistent",
    num_workers: int = 1,
    target_sequence: str = "flair",
    guidance_sequences: Sequence[str] = ("t1", "t2"),
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    random_slicing_direction: bool = False,
    deregister_images: bool = False,
    wmh_mask: bool = False,
    sr_input: float = 0.0,
    aniso_t2: bool = False,
    is_train: bool = True,
):
    relevant_sequences = [target_sequence] + list(guidance_sequences)

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augs = (
        [
            transforms.SpatialPadd(
                keys=relevant_sequences + ["mask", "brain_mask"],
                spatial_size=(128, -1, -1),
                allow_missing_keys=True,
            ),
            transforms.RandSpatialCropD(
                keys=relevant_sequences + ["mask", "brain_mask"],
                roi_size=roi_size,
                random_center=True,
            ),
        ]
        if is_train
        else []
    )

    train, val = get_datasets(
        dataset=dataset,
        data_dir=data_dir,
        relevant_sequences=list(relevant_sequences),
        normalize_to=(-1, 1),
        size=img_size,
        skull_strip=skull_strip,
        cache=cache,
        subset_val=subset_val,
        subset_train=subset_train,
        aug_transforms=transforms.Compose(augs),
        crop_to_brain_margin=None,
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    print(f"Rank {os.getenv('LOCAL_RANK')} - Train size: {len(train)}")
    return train_loader, val_loader
