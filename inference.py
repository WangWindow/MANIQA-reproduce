import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.config import Config, setup_seed
from utils.inference_process import Normalize, ToTensor, eval_epoch, sort_file
from utils.PIPAL22.pipal22_test import PIPAL22

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config(
        {
            # dataset path
            "db_name": "PIPAL",
            "test_dis_path": "/mnt/data_16TB/ysd21/IQA/NTIRE2022_NR_Valid_Dis/",
            # optimization
            "batch_size": 10,
            "num_avg_val": 1,
            "crop_size": 224,
            # device
            "num_workers": 8,
            # load & save checkpoint
            "valid": "./checkpoints/valid",
            "valid_path": "./checkpoints/valid/inference_valid",
            "model_path": "./checkpoints/models/model_maniqa/epoch1",
        }
    )

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)

    # data load
    test_dataset = PIPAL22(
        dis_path=config.test_dis_path,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False,
    )
    net = torch.load(config.model_path)
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + "/output.txt")
