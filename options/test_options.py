from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            "--ntest", type=int, default=float("inf"), help="# of test examples."
        )
        self.parser.add_argument(
            "--results_dir", type=str, default="./results/", help="saves results here."
        )
        self.parser.add_argument(
            "--aspect_ratio",
            type=float,
            default=1.0,
            help="aspect ratio of result images",
        )
        self.parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        self.parser.add_argument(
            "--dataset_json",
            type=str,
            default="../data/RS/RS_train_split.json",
            help="path to the dataset json file",
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="../data/RS/conformed_mask_reg",
            help="path to the dataset directory",
        )
        self.parser.add_argument(
            "-o",
            "--out_dir_name",
            type=str,
            default="inference",
            help="output directory name for the test images",
        )
        self.isTrain = False
