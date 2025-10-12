#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from MPNCOV.python import MPNCOV
from srm_filter_kernel import all_normalized_hpf_list


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_array = np.stack(all_hpf_list_5x5)
        hpf_weight = nn.Parameter(torch.from_numpy(hpf_array).float().view(30, 1, 5, 5),
                                  requires_grad=False)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)
        return output
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output
class StegFormerNet(nn.Module):

  def __init__(self):
    super(StegFormerNet, self).__init__()

    self.group1 = HPF()

    self.group2 = nn.Sequential(
      nn.Conv2d(30, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
      # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    )

    self.group3 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
      # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    )

    self.group4 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
      # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    )

    self.group5 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      #nn.AvgPool2d(kernel_size=32, stride=1)
    )

    self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)
    #self.fc1 = nn.Linear(1 * 1 * 256, 2)

  def forward(self, input):
    output = input

    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)

    output = MPNCOV.CovpoolLayer(output)
    output = MPNCOV.SqrtmLayer(output, 5)
    output = MPNCOV.TriuvecLayer(output)

    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output
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
class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                 transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform

        self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
        self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.pgm'

        self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
        self.bows_stego_path = BOWS_STEGO_DIR + '/{}.pgm'
        # 在 MyDataset.__init__ 中添加
        print(f"Dataset index path: {index_path}")
        print(f"Number of samples: {len(self.index_list)}")

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        if file_index <= 10000:
            cover_path = self.bossbase_cover_path.format(file_index)
            stego_path = self.bossbase_stego_path.format(file_index)
        else:
            cover_path = self.bows_cover_path.format(file_index - 10000)
            stego_path = self.bows_stego_path.format(file_index - 10000)

        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)
        '''
        cover_data = sio.loadmat(cover_path)['img_mat']
        stego_data = sio.loadmat(stego_path)['img_mat']
        '''
        data = np.stack([cover_data, stego_data])

        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        # 可视化检查
        import matplotlib.pyplot as plt

        # 随机选择几个样本检查
        if idx < 3:  # 检查前3个样本
            sample = {'data': data, 'label': label}
            print(f"Sample {idx}: Label={label}")
            plt.imshow(data[0], cmap='gray')  # 显示cover图像
            plt.title(f"Cover (Label=0)")
            plt.show()
            plt.imshow(data[1], cmap='gray')  # 显示stego图像
            plt.title(f"Stego (Label=1)")
            plt.show()


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            # 处理数据形状
            if len(data.shape) == 5:  # [batch, 2, C, H, W]
                data = data.reshape(-1, *data.shape[2:])
                label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)

    accuracy = correct / total if total > 0 else 0

    if accuracy > best_acc and epoch > 10:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    print('-' * 8)
    print(f'Eval accuracy: {accuracy:.4f}')
    print(f'Eval err: {1 - accuracy:.4f}')
    print(f'Best accuracy: {best_acc:.4f}')
    print('-' * 8)

    return accuracy


# 隐写算法列表
STEGO_ALGORITHMS = ['hugo', 'wow', 'suniward', 'mg', 'mipod', 'synch']
OUTPUT_PATH = Path(__file__).stem
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_model(model_path, device):
    """加载训练好的HybridModel"""
    model = StegFormerNet().to(device)
    state = torch.load(model_path)
    model.load_state_dict(state['original_state'])
    model.eval()
    return model


def evaluate_mismatch(model, device, cover_dir, stego_dir, index_path):
    """评估模型在特定数据集上的表现"""
    dataset = MyDataset(
        index_path=index_path,
        BOSSBASE_COVER_DIR=cover_dir,
        BOSSBASE_STEGO_DIR=stego_dir,
        BOWS_COVER_DIR='',
        BOWS_STEGO_DIR='',
        transform=transforms.Compose([ToTensor()])
    )

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = sample['data'].to(device)
            label = sample['label'].to(device)

            # 调试信息：检查数据形状
            if i == 0:
                print(f"Batch {i}: data shape = {data.shape}, label shape = {label.shape}")

            # 处理成对数据 [batch, 2, channel, height, width] -> [batch*2, channel, height, width]
            if len(data.shape) == 5:
                data = data.reshape(-1, *data.shape[2:])
                label = label.reshape(-1)
                if i == 0:
                    print(f"After reshape: data shape = {data.shape}, label shape = {label.shape}")

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)

            # 调试信息：检查模型输出
            if i == 0:
                print(f"Model output shape: {output.shape}")
                print(f"Sample predictions: {pred[:10].flatten()}")
                print(f"Sample labels: {label[:10]}")

    accuracy = correct / total
    print(f"Final accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def test_all_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).parent

    # 准备数据集路径
    bossbase_cover = str(project_root / 'data' / 'BossBase-256')
    bows_cover = str(project_root / 'data' / 'BOWS2-256')
    test_index = str(project_root / 'index_list' / 'bossbase_test_index.npy')

    # 初始化结果表
    bossbase_results = pd.DataFrame(index=STEGO_ALGORITHMS, columns=STEGO_ALGORITHMS)
    bows_results = pd.DataFrame(index=STEGO_ALGORITHMS, columns=STEGO_ALGORITHMS)

    # 测试每个训练好的模型
    for train_algo in STEGO_ALGORITHMS:
        # BossBase模型
        model_path = os.path.join(OUTPUT_PATH,
                                  f'bossbase-{train_algo}-model.pt')
        if os.path.exists(model_path):
            print(f"\nTesting BossBase model trained on {train_algo}...")
            model = load_model(model_path, device)

            for test_algo in STEGO_ALGORITHMS:
                stego_dir = str(project_root / 'data' / f'BossBase-{test_algo}')
                acc = evaluate_mismatch(model, device, bossbase_cover, stego_dir, test_index)
                bossbase_results.loc[train_algo, test_algo] = f"{acc:.4f}"

        # BOWS2模型
        model_path = os.path.join(OUTPUT_PATH,
                                  f'bows-{train_algo}-model.pt')
        if os.path.exists(model_path):
            print(f"\nTesting BOWS2 model trained on {train_algo}...")
            model = load_model(model_path, device)

            for test_algo in STEGO_ALGORITHMS:
                stego_dir = str(project_root / 'data' / f'BOWS2-{test_algo}')
                acc = evaluate_mismatch(model, device, bows_cover, stego_dir, test_index)
                bows_results.loc[train_algo, test_algo] = f"{acc:.4f}"

    # 创建样式化的表格输出函数
    def print_styled_table(df, title):
        # 设置列宽和居中
        col_width = 10
        center_fmt = lambda x: f"{x:^{col_width}}"

        # 构建表头
        header = "TRN\\TST".ljust(col_width) + "|" + "|".join(
            center_fmt(col) for col in df.columns
        )

        # 构建分隔线
        separator = "-" * len(header)

        # 打印标题和表头
        print("\n" + "=" * len(header))
        print(title.center(len(header)))
        print("=" * len(header))
        print(header)
        print(separator)

        # 打印每行数据
        for idx, row in df.iterrows():
            row_str = idx.ljust(col_width) + "|" + "|".join(
                center_fmt(val) for val in row
            )
            print(row_str)

        print("=" * len(header) + "\n")

    # 打印BossBase结果
    print_styled_table(
        bossbase_results,
        "StegFormer-Net Model - BossBase Dataset (Embedding Rate: 0.4bpp)"
    )

    # 打印BOWS2结果
    print_styled_table(
        bows_results,
        "StegFormer-Net Model - BOWS2 Dataset (Embedding Rate: 0.4bpp)"
    )

    # 保存结果到CSV
    bossbase_results.to_csv(os.path.join(OUTPUT_PATH, 'bossbase_test_results.csv'))
    bows_results.to_csv(os.path.join(OUTPUT_PATH, 'bows2_test_results.csv'))

    return bossbase_results, bows_results


def print_results(df, title):
    """美化打印结果表格"""
    print(f"\n=== {title} ===")
    print(df.to_string(float_format="%.4f"))


def main(args):
    if args.mode == 'test':
        test_all_models(args)
    else:
        # 训练逻辑（可以保留或移除）
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
