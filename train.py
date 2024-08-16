import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import sys

sys.path.append(os.path.realpath(os.path.dirname(os.getcwd())))

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("#training images = %d" % dataset_size)

model = create_model(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
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
            "saving the model at the end of epoch %d, iters %d" % (epoch, total_steps)
        )
        model.save("latest")
        model.save(epoch)

    print(
        "End of epoch %d / %d \t Time Taken: %d sec"
        % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
    )

    if opt.rise_sobelLoss and epoch <= 20:
        model.update_sobel_lambda(epoch)

    if epoch > opt.niter:
        model.update_learning_rate()
