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
BOSS_MODEL_DIR = 'yednet_bossbase'
BOWS_MODEL_DIR = 'yednet_bows'
OUTPUT_PATH = Path(__file__).stem
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ========== YedNet模型定义 ==========
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output


# add operation
class ADD(nn.Module):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, input1, input2):
        output = torch.add(input1, input2)
        return output


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5),
                                  requires_grad=False)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        return self.tlu(output)


class YedNet(torch.nn.Module):
    def __init__(self):
        super(YedNet, self).__init__()
        self.group1 = HPF()  # pre-processing Layer 1
        # self.conv0.weight = torch.nn.Parameter(srm)

        self.conv1 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=1,
                                     padding=2)  # Sepconv Block 1 Layer 2
        self.abs = ABS()
        self.bn1 = nn.BatchNorm2d(30)
        # Trunc T= 3
        self.tlu3 = TLU(3.0)
        self.conv2 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=1,
                                     padding=2)  # Sepconv Block 2 Layer 3
        self.bn2 = nn.BatchNorm2d(30)
        # Trunc T = 1
        self.tlu1 = TLU(1.0)
        self.pool = torch.nn.AvgPool2d(kernel_size=5, stride=2,
                                       padding=2)  # the same pool layer well be used to L3 and L4
        self.conv3 = torch.nn.Conv2d(in_channels=30, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 5
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 6
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 7
        self.bn7 = nn.BatchNorm2d(128)

        # self.spp_layer = SPPLayer(spp_level) # spp_level = 1 Global averge pooling

        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)

    def forward(self, x):
        x = self.group1(x)
        x = self.conv1(x)
        x = self.abs(x)
        # x =  F.relu(x)
        x = self.bn1(x)
        x = self.tlu3(x)

        x = self.bn2(self.conv2(x))
        x = self.tlu1(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return (x)


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


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[1, 2]).copy()

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


# ========== 评估函数 ==========
def evaluate(model, device, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].view(-1, 1, 256, 256).to(device)
            label = sample['label'].view(-1).to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
    return correct / (len(data_loader.dataset) * 2)


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
            print(f"\n=== Testing BossBase Model [{train_algo}] ===")
            model = YedNet().to(device)
            state = torch.load(boss_model_path, weights_only=True)
            model.load_state_dict(state['original_state'])
            print(model)  # 打印完整模型结构
            print("First conv weight mean:", model.conv1.weight.mean().item())  # 检查权重是否加载

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
            print(f"\n=== Testing BOWS Model [{train_algo}] ===")
            model = YedNet().to(device)
            state = torch.load(bows_model_path, weights_only=True)
            model.load_state_dict(state['original_state'])
            print(model)  # 打印完整模型结构
            print("First conv weight mean:", model.conv1.weight.mean().item())  # 检查权重是否加载

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
        print(f"\n{'=' * 80}")
        print(f"{'TRN\\TST':<10}" + "".join(f"{alg:>10}" for alg in STEGO_ALGORITHMS))
        print('-' * 80)
        for trn in STEGO_ALGORITHMS:
            print(f"{trn:<10}", end="")
            for tst in STEGO_ALGORITHMS:
                print(f"{df.loc[trn, tst]:>10}", end="")
            print()
        print('=' * 80)

    print_table(boss_results, "YedNet Model - BossBase Dataset (Embedding Rate: 0.4bpp)")
    print_table(bows_results, "YedNet Model - BOWS2 Dataset (Embedding Rate: 0.4bpp")

    # 保存结果
    boss_results.to_csv(os.path.join(OUTPUT_PATH, 'yednet_bossbase_results.csv'))
    bows_results.to_csv(os.path.join(OUTPUT_PATH, 'yednet_bows_results.csv'))

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
