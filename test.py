import os, sys
from options.test_options import TestOptions
from models.models import create_model
from omegaconf import OmegaConf
from data import create_loaders
from tqdm import tqdm
import torch
from flairsyn.lib.utils.visualization import vol_view
from flairsyn.lib.inference import save_output_volume
from monai.transforms import CropForegroundd
from flairsyn.lib.datasets import get_datasets
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def main(opt):
    output_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.out_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions to: {output_dir}")
    config = OmegaConf.load(opt.config)
    relevant_sequences = [config.data.target_sequence] + [
        *config.data.guidance_sequences
    ]
    operating_size = opt.operating_size
    print(f"Operating size: {operating_size}")

    _, val = get_datasets(
        dataset=opt.dataset_json,
        data_dir=opt.data_dir,
        relevant_sequences=relevant_sequences,
        size=None,
        cache=None,
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=config.data.skull_strip and not opt.no_skull_strip,
    )

    model = create_model(opt)
    model.netG.eval().cuda()

    # crop_t = CropForegroundd(
    #     keys=["t1", "t2", "flair", "pred", "mask"],
    #     source_key="mask",
    #     allow_smaller=True,
    #     margin=(5, 10, 20),  # as D H W
    #     allow_missing=True,
    # )

    progress_bar = tqdm(enumerate(val), total=len(val))
    with torch.no_grad():
        for i, data in progress_bar:
            input = torch.cat(
                [data[seq] for seq in config.data.guidance_sequences], dim=0
            )

            # print(f"Processing {data['subject_ID']}, input shape: {input.shape}")

            # pad to operating size
            pad = -torch.ones(input.shape[0], *operating_size)
            offset_d = (operating_size[0] - input.shape[1]) // 2
            offset_h = (operating_size[1] - input.shape[2]) // 2
            offset_w = (operating_size[2] - input.shape[3]) // 2
            assert offset_d >= 0, f"input too large: {input.shape}, {operating_size}"
            assert offset_h >= 0, f"input too large: {input.shape}, {operating_size}"
            assert offset_w >= 0, f"input too large: {input.shape}, {operating_size}"
            pad[
                :,
                offset_d : offset_d + input.shape[1],
                offset_h : offset_h + input.shape[2],
                offset_w : offset_w + input.shape[3],
            ] = input

            # divide image into 8 blocks with 128 x 128 x 128
            batch = [
                pad[:, 0:128, 0:128, 0:128],
                pad[:, 0:128, 0:128, -128:],
                pad[:, 0:128, -128:, 0:128],
                pad[:, 0:128, -128:, -128:],
                pad[:, -128:, 0:128, 0:128],
                pad[:, -128:, 0:128, -128:],
                pad[:, -128:, -128:, 0:128],
                pad[:, -128:, -128:, -128:],
            ]
            guid = torch.stack(batch, dim=0).float().cuda()
            output = model.netG(guid)

            # output = (output + 1).cpu().squeeze(dim=1) / 2
            output = output.cpu().squeeze(dim=1)
            res = torch.zeros(operating_size)
            weights = torch.zeros(operating_size)

            res[0:128, 0:128, 0:128] += output[0]
            weights[0:128, 0:128, 0:128] += 1
            res[0:128, 0:128, -128:] += output[1]
            weights[0:128, 0:128, -128:] += 1
            res[0:128, -128:, 0:128] += output[2]
            weights[0:128, -128:, 0:128] += 1
            res[0:128, -128:, -128:] += output[3]
            weights[0:128, -128:, -128:] += 1
            res[-128:, 0:128, 0:128] += output[4]
            weights[-128:, 0:128, 0:128] += 1
            res[-128:, 0:128, -128:] += output[5]
            weights[-128:, 0:128, -128:] += 1
            res[-128:, -128:, 0:128] += output[6]
            weights[-128:, -128:, 0:128] += 1
            res[-128:, -128:, -128:] += output[7]
            weights[-128:, -128:, -128:] += 1
            res /= weights

            # crop to original size
            res = res[
                offset_d : offset_d + data["t1"].shape[1],
                offset_h : offset_h + data["t1"].shape[2],
                offset_w : offset_w + data["t1"].shape[3],
            ]

            data["pred"] = res.unsqueeze(0)
            # data = crop_t(data)

            save_output_volume(
                data,
                output_path=output_dir,
                save_keys=relevant_sequences + ["pred", "mask"],
                target_sequence=config.data.target_sequence,
            )


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.serial_batches = True
    opt.no_flip = True
    main(opt)
