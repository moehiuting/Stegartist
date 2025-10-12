#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import random
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from srm_filter_kernel import all_normalized_hpf_list
import torch.nn.functional as F

# 隐写算法列表
STEGO_ALGORITHMS = ['hugo', 'wow', 'suniward', 'mg', 'mipod', 'synch']

# 模型文件夹
BOSS_MODEL_DIR = 'xunet_train_bossbase'
BOWS_MODEL_DIR = 'xunet_train_bows'
OUTPUT_PATH = Path(__file__).stem
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ========== XUNet模型定义 ==========
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        return torch.clamp(input, min=-self.threshold, max=self.threshold)


class AbsWrapper(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class HPF_srm6(nn.Module):
    """6个SRM高通滤波器"""

    def __init__(self):
        super(HPF_srm6, self).__init__()
        # 选择前6个SRM滤波器
        selected_filters = all_normalized_hpf_list[:6]
        all_hpf_list_5x5 = []

        for hpf_item in selected_filters:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_array = np.stack(all_hpf_list_5x5)
        hpf_weight = nn.Parameter(torch.from_numpy(hpf_array).float().view(6, 1, 5, 5),
                                  requires_grad=False)
        self.hpf = nn.Conv2d(1, 6, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(3.0)

    def forward(self, x):
        x = self.hpf(x)
        x = self.tlu(x)
        return x


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        self.hpf = HPF_srm6()

        self.group1 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=2, bias=False),
            AbsWrapper(),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=16, stride=16)  # 全局平均池化
        )
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.hpf(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# ========== 数据加载 ==========
class MyDataset(Dataset):
    def __init__(self, index_path, cover_dir, stego_dir, transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform
        self.cover_path = os.path.join(cover_dir, '{}.pgm')
        self.stego_path = os.path.join(stego_dir, '{}.pgm')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_index = self.index_list[idx]
        cover = cv2.imread(self.cover_path.format(file_index), -1)
        stego = cv2.imread(self.stego_path.format(file_index), -1)

        if cover is None or stego is None:
            raise ValueError(f"Failed to load image {file_index}")

        data = np.stack([cover, stego])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # 确保数据格式正确
        if data.ndim == 3:  # [2, H, W]
            data = np.expand_dims(data, axis=1)  # [2, 1, H, W]

        data = data.astype(np.float32)
        data = data / 255.0  # 归一化

        return {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }


# ========== 评估函数 ==========
def evaluate(model, device, data_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in data_loader:
            data = sample['data']
            label = sample['label']

            # 处理成对数据 [batch, 2, 1, H, W] -> [batch*2, 1, H, W]
            data = data.view(-1, 1, 256, 256).to(device)
            label = label.view(-1).to(device)

            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += label.size(0)

    return correct / total if total > 0 else 0


# ========== 测试函数 ==========
def test_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据集路径
    project_root = Path(__file__).parent
    boss_cover = str(project_root / 'data' / 'BossBase-256')
    bows_cover = str(project_root / 'data' / 'BOWS2-256')
    test_index = str(project_root / 'index_list' / 'bossbase_test_index.npy')

    # 创建结果表格
    boss_results = pd.DataFrame(index=STEGO_ALGORITHMS, columns=STEGO_ALGORITHMS)
    bows_results = pd.DataFrame(index=STEGO_ALGORITHMS, columns=STEGO_ALGORITHMS)

    # 数据转换
    transform = transforms.Compose([ToTensor()])

    # 测试每个模型
    for train_algo in STEGO_ALGORITHMS:
        # 测试BossBase训练的模型
        boss_model_path = os.path.join(BOSS_MODEL_DIR, f'{train_algo}-0.4-1-0.50-params.pt')
        if os.path.exists(boss_model_path):
            print(f"\n=== Testing BossBase XUNet Model [{train_algo}] ===")
            model = XuNet().to(device)
            state = torch.load(boss_model_path, map_location=device)
            model.load_state_dict(state['original_state'])
            model.eval()

            for test_algo in STEGO_ALGORITHMS:
                print(f"Testing against {test_algo}...")
                stego_dir = str(project_root / 'data' / f'BossBase-{test_algo}')
                dataset = MyDataset(test_index, boss_cover, stego_dir, transform)
                loader = DataLoader(dataset, batch_size=16, shuffle=False)
                acc = evaluate(model, device, loader)
                boss_results.loc[train_algo, test_algo] = f'{acc:.4f}'

        # 测试BOWS训练的模型
        bows_model_path = os.path.join(BOWS_MODEL_DIR, f'{train_algo}-0.4-1-0.50-params.pt')
        if os.path.exists(bows_model_path):
            print(f"\n=== Testing BOWS XUNet Model [{train_algo}] ===")
            model = XuNet().to(device)
            state = torch.load(bows_model_path, map_location=device)
            model.load_state_dict(state['original_state'])
            model.eval()

            for test_algo in STEGO_ALGORITHMS:
                print(f"Testing against {test_algo}...")
                stego_dir = str(project_root / 'data' / f'BOWS2-{test_algo}')
                dataset = MyDataset(test_index, bows_cover, stego_dir, transform)
                loader = DataLoader(dataset, batch_size=16, shuffle=False)
                acc = evaluate(model, device, loader)
                bows_results.loc[train_algo, test_algo] = f'{acc:.4f}'

    # 打印美观的表格
    def print_table(df, title):
        print(f"\n{'=' * 80}")
        print(f"{title.center(80)}")
        print(f"{'=' * 80}")
        print(f"{'TRN\\TST':<10}" + "".join(f"{alg:>10}" for alg in STEGO_ALGORITHMS))
        print('-' * 80)
        for trn in STEGO_ALGORITHMS:
            print(f"{trn:<10}", end="")
            for tst in STEGO_ALGORITHMS:
                print(f"{df.loc[trn, tst]:>10}", end="")
            print()
        print('=' * 80)

    print_table(boss_results, "XUNet Model - BossBase Dataset (Embedding Rate: 0.4bpp)")
    print_table(bows_results, "XUNet Model - BOWS2 Dataset (Embedding Rate: 0.4bpp)")

    # 保存结果
    boss_results.to_csv(os.path.join(OUTPUT_PATH, 'xunet_bossbase_results.csv'))
    bows_results.to_csv(os.path.join(OUTPUT_PATH, 'xunet_bows_results.csv'))

    return boss_results, bows_results


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, choices=['0', '1', '2', '3'],
                        required=True, help='GPU to use')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test_models(args)


if __name__ == '__main__':
    main()
