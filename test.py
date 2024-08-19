import monai.data.meta_tensor
import time
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


def main(opt):
    output_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.out_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions to: {output_dir}")

    _, val = get_datasets(
        dataset=opt.dataset_json,
        data_dir=opt.data_dir,
        relevant_sequences=["flair", "t1", "t2"],
        size=(240, 240, 176),  # fix size, crop afterward
        cache=None,
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=1,
    )

    model = create_model(opt)
    model.netG.eval().cuda()

    crop_t = CropForegroundd(
        keys=["t1", "t2", "flair", "pred", "mask"],
        source_key="mask",
        allow_smaller=True,
        margin=(5, 10, 20),  # as D H W
        allow_missing=True,
    )

    progress_bar = tqdm(enumerate(val), total=len(val))
    with torch.no_grad():
        for i, data in progress_bar:
            input = torch.cat([data["t1"], data["t2"]], dim=0)
            # divide image into 8 blocks with 128 x 128 x 128
            batch = [
                input[:, 0:128, 0:128, 0:128],
                input[:, 0:128, 0:128, -128:],
                input[:, 0:128, -128:, 0:128],
                input[:, 0:128, -128:, -128:],
                input[:, -128:, 0:128, 0:128],
                input[:, -128:, 0:128, -128:],
                input[:, -128:, -128:, 0:128],
                input[:, -128:, -128:, -128:],
            ]
            output = model.netG(torch.stack(batch, dim=0).float().cuda())

            # output = (output + 1).cpu().squeeze(dim=1) / 2
            output = output.cpu().squeeze(dim=1)
            res = torch.zeros(data["t1"].shape[1:])
            weights = torch.zeros(data["t1"].shape[1:])

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

            data["pred"] = res.unsqueeze(0)
            data = crop_t(data)

            save_output_volume(
                data,
                output_path=output_dir,
                save_keys=["t1", "t2", "flair", "pred", "mask"],
                target_sequence="flair",
            )


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.serial_batches = True
    opt.no_flip = True
    main(opt)
