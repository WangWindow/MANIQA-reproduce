import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from backbones.maniqa import MANIQA
from utils.config import Config, setup_seed
from utils.inference_process import Normalize, ToTensor, eval_epoch, sort_file

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
            # dataset
            "dataset_name": "roi",
            "roi_base_dir": "./data/ROI_Data/",
            "roi_label": "./utils/roi_dataset/sdd_quality_pseudo_labels.txt",
            # optimization
            "batch_size": 10,
            "num_avg_val": 1,
            "crop_size": 224,
            # device
            "num_workers": 8,
            # load & save checkpoint
            "valid": "./checkpoints/valid",
            "valid_path": "./checkpoints/valid/inference_valid",
            # NOTE: should point to a .pt saved by train.py (state_dict)
            "ckpt_path": "./checkpoints/models/ROI/roi_s20/latest.pt",
            # model
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,
        }
    )

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)

    # data load
    if config.dataset_name == "roi":
        from utils.roi_dataset.roi_dataset import ROIDatasetInference

        test_dataset = ROIDatasetInference(
            base_dir=config.roi_base_dir,
            txt_file_name=config.roi_label,
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            img_size=config.img_size,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
    )

    net = MANIQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale,
    )
    state_dict = torch.load(config.ckpt_path, map_location="cpu")
    net.load_state_dict(state_dict, strict=False)
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + "/output.txt")
