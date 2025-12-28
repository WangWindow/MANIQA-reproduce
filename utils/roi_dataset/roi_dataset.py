import os

import cv2
import numpy as np
import torch


def _safe_minmax_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    rng = vmax - vmin
    if rng <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / rng


def _resolve_path(base_dir: str, path_str: str) -> str:
    path_str = path_str.strip()
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_dir, path_str))


class ROIDataset(torch.utils.data.Dataset):
    """ROI 数据集（监督训练用）。

    读取 sdd-fiqa 生成的伪标签文件：每行 `image_path<TAB>score`。
    返回 sample: {"d_img_org": CHW float32, "score": (1,)}
    """

    def __init__(
        self,
        base_dir: str,
        txt_file_name: str,
        list_name,
        transform,
        keep_ratio: float = 1.0,
        img_size: int = 224,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.img_size = img_size

        list_name = set(list_name) if list_name is not None else None

        img_list, score_list = [], []
        with open(self.txt_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                path_str, score_str = parts[0], parts[1]
                if list_name is not None and path_str not in list_name:
                    continue
                img_list.append(path_str)
                score_list.append(float(score_str))

        if keep_ratio < 1.0:
            keep_n = max(1, int(len(img_list) * keep_ratio))
            img_list = img_list[:keep_n]
            score_list = score_list[:keep_n]

        score_arr = _safe_minmax_norm(np.array(score_list, dtype=np.float32))
        score_arr = score_arr.astype("float32").reshape(-1, 1)

        self.data_dict = {"d_img_list": img_list, "score_list": score_arr}

    def __len__(self):
        return len(self.data_dict["d_img_list"])

    def __getitem__(self, index):
        rel_path = self.data_dict["d_img_list"][index]
        img_path = _resolve_path(self.base_dir, rel_path)

        d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if d_img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        d_img = cv2.resize(
            d_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255.0
        d_img = np.transpose(d_img, (2, 0, 1))

        score = self.data_dict["score_list"][index]
        sample = {"d_img_org": d_img, "score": score}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ROIDatasetInference(torch.utils.data.Dataset):
    """ROI 数据集（推理用）。

    默认从伪标签文件中取图片路径（忽略分数），返回 {"d_img_org", "d_name"}。
    """

    def __init__(
        self,
        base_dir: str,
        txt_file_name: str,
        transform,
        img_size: int = 224,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.img_size = img_size

        img_list = []
        with open(self.txt_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                img_list.append(parts[0])

        self.data_dict = {"d_img_list": img_list}

    def __len__(self):
        return len(self.data_dict["d_img_list"])

    def __getitem__(self, index):
        rel_path = self.data_dict["d_img_list"][index]
        img_path = _resolve_path(self.base_dir, rel_path)

        d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if d_img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        d_img = cv2.resize(
            d_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype("float32") / 255.0
        d_img = np.transpose(d_img, (2, 0, 1))

        sample = {"d_img_org": d_img, "d_name": rel_path}
        if self.transform:
            sample = self.transform(sample)
        return sample
