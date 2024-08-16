import time
from options.train_options import TrainOptions
from data import create_loaders
from models.models import create_model
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# from image_similarity_measures.quality_metrics import ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from omegaconf import OmegaConf
import torch

sys.path.append(os.path.realpath(os.path.dirname(os.getcwd())))
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def main(opt):
    torch.multiprocessing.set_sharing_strategy("file_system")
    config = OmegaConf.load("config.yml")
    data_loader, _ = create_loaders(
        batch_size=opt.batchSize, roi_size=opt.fineSize, **config.data
    )
    dataset_size = len(data_loader)
    print("#training images = %d" % dataset_size)

    model = create_model(opt)
    out_path = os.path.join(opt.checkpoints_dir, opt.name)
    writer = SummaryWriter(log_dir=os.path.join(out_path, "tb"))

    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in progress_bar:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)

            # I have no idea why implement it like this
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_steps %d)"
                    % (epoch, total_steps)
                )
                model.save("latest")

        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_steps)
            )
            model.save("latest")
            model.save(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )

        # writer.add_scalar("L1_val", l1_avg_loss, epoch)
        # writer.add_scalar("PSNR", mean_psnr, epoch)
        # writer.add_scalar("SSIM", np.mean(ssim_avg[epoch - 1]), epoch)

        if opt.rise_sobelLoss and epoch <= 20:
            model.update_sobel_lambda(epoch)

        if epoch > opt.niter:
            model.update_learning_rate()


if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)
