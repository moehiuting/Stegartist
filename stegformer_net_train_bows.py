#!/usr/bin/env python3
import math
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import logging
import random
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from srm_filter_kernel import all_normalized_hpf_list
# from srm_filter_kernel import all_hpf_list
from MPNCOV.python import MPNCOV

PROP = 0.50
IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
# EPOCHS = 8
# LR = 0.4
LR = 0.01
WEIGHT_DECAY = 5e-4

# EMBEDDING_RATE = 0.2

# LOG_INTERATION_INTERVAL = 100
# # TRAIN_FILE_COUNT = 4000
# TEST_INTERATION_INTERVAL = 1000

TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]


# Transformer hyperparameters
NUM_HEADS = 4
NUM_LAYERS = 4
PROJECTION_DIM = 128
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
PATCH_SIZE = 4
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

OUTPUT_PATH = Path(__file__).stem
try:
    os.makedirs(OUTPUT_PATH)
except OSError:
    pass


def acc_plot(hist, path='', model_name=''):
    x = range(len(hist['acc']))
    y1 = hist['acc']
    y2 = hist['err']

    plt.plot(x, y1, label='acc')
    plt.plot(x, y2, label='err')

    plt.xlabel('Iter')
    plt.ylabel('Acc')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'acc_' + model_name + '.png')

    plt.savefig(path)

    plt.close()


# 定义TLU激活函数
class TLU(nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.tanh(x) * self.threshold


# SRM滤波器层
class SRMLayer(nn.Module):
    def __init__(self, srm_weights):
        super().__init__()
        # 固定分支（预训练SRM滤波器）
        self.conv_fixed = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=True)
        # 可训练分支（独立初始化）
        self.conv_train = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=True)

        # 最大值归一化初始化
        with torch.no_grad():
            srm_weights_pt = torch.from_numpy(srm_weights).float().permute(3, 2, 0, 1)
            max_val = torch.max(torch.abs(srm_weights_pt))
            srm_weights_pt = srm_weights_pt / max_val

            self.conv_fixed.weight.data = srm_weights_pt
            self.conv_fixed.bias.data = torch.ones(30)

            nn.init.kaiming_normal_(self.conv_train.weight, mode='fan_out', nonlinearity='linear')
            max_val = torch.max(torch.abs(self.conv_train.weight))
            self.conv_train.weight.data = self.conv_train.weight / max_val

        # 固定固定分支的参数
        for param in self.conv_fixed.parameters():
            param.requires_grad = False

        self.activation = nn.Tanh()
        self.scale = 3.0
        self.bn = nn.BatchNorm2d(30, momentum=0.2, eps=0.001, affine=True)

    def forward(self, x):
        with torch.no_grad():
            fixed = self.conv_fixed(x)
        train = self.conv_train(x)
        out = fixed + train
        out = self.activation(out) * self.scale
        out = self.bn(out)
        return out


# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16, conv=False):
        super().__init__()
        self.conv = conv
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

        if conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        scale = x * y
        shortcut = self.shortcut(x)
        out = shortcut + scale
        return out


# 基础卷积块
class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 带SE注意力的残差块
class Block2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return out


# 下采样残差块
class Block3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return out


# Transformer模块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerModule(nn.Module):
    def __init__(self, in_channels=256, patch_size=4, dim=128, depth=4, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.se = SEBlock(dim, ratio=32)
        self.pos_embed = PositionalEncoding(dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.se(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)


# 完整模型
class SteganalysisModel(nn.Module):
    def __init__(self, srm_weights):
        super().__init__()
        self.preprocess = SRMLayer(srm_weights)

        self.block1 = Block1(30, 64)
        self.block2 = Block1(64, 64)
        self.blocks = nn.Sequential(
            *[Block2(64) for _ in range(5)],
            Block3(64, 64),
            Block3(64, 64),
            Block3(64, 128),
            Block3(128, 256)
        )

        self.transformer = TransformerModule(
            in_channels=256,
            patch_size=4,
            dim=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch_size, num_pairs = x.shape[:2]
        x = x.view(-1, *x.shape[2:])

        x = self.preprocess(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.blocks(x)
        x = self.transformer(x)
        x = self.classifier(x)

        x = x.view(batch_size, num_pairs, -1)
        return torch.sigmoid(x)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()  # ONE EPOCH TRAIN TIME
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        # 修改reshape方式
        batch_size = data.size(0)
        data = data.view(batch_size * 2, 1, 256, 256)  # [batch*2, 1, H, W]
        label = label.view(-1)  # [batch*2]

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            # 保持与train相同的reshape
            batch_size = data.size(0)
            data = data.view(batch_size * 2, 1, 256, 256)
            label = label.view(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            batch_size = data.size(0)
            data = data.view(batch_size * 2, 1, 256, 256)  # [batch*2, 1, H, W]
            label = label.view(-1)  # [batch*2]

            data, label = data.to(device), label.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    if accuracy > best_acc and epoch > 10:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Eval err: {:.4f}'.format(1 - accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)

    return accuracy


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

            # nn.init.xavier_uniform_(module.weight.data)
            # nn.init.constant_(module.bias.data, val=0.2)
        # else:
        #   module.weight.requires_grad = True

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[1, 2]).copy()
        # for i in range(0,rot):
        #  data = np.rollaxis(data,1,2).copy()

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # 确保数据是float32类型
        data = data.astype(np.float32)

        # 添加通道维度 (2, H, W) -> (2, 1, H, W)
        data = np.expand_dims(data, axis=1)

        # 转换为torch张量
        new_sample = {
            'data': torch.from_numpy(data),  # 最终形状: [2, 1, 256, 256]
            'label': torch.from_numpy(label).long(),
        }

        # 调试信息
        print(f"ToTensor output shape: {new_sample['data'].shape}")  # 应输出: torch.Size([2, 1, 256, 256])

        return new_sample


class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                 transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform
        self.BOSSBASE_COVER_DIR = BOSSBASE_COVER_DIR
        self.BOSSBASE_STEGO_DIR = BOSSBASE_STEGO_DIR
        self.BOWS_COVER_DIR = BOWS_COVER_DIR
        self.BOWS_STEGO_DIR = BOWS_STEGO_DIR

        print(f"Dataset index path: {index_path}")
        print(f"Number of samples: {len(self.index_list)}")

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        try:
            if file_index <= 10000:
                cover_path = os.path.join(self.BOSSBASE_COVER_DIR, f"{file_index}.pgm")
                stego_path = os.path.join(self.BOSSBASE_STEGO_DIR, f"{file_index}.pgm")
            else:
                cover_path = os.path.join(self.BOWS_COVER_DIR, f"{file_index - 10000}.pgm")
                stego_path = os.path.join(self.BOWS_STEGO_DIR, f"{file_index - 10000}.pgm")

            # 确保以灰度模式加载图像
            cover_data = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_data = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            # 检查图像是否成功加载
            if cover_data is None:
                raise ValueError(f"Failed to load cover image: {cover_path}")
            if stego_data is None:
                raise ValueError(f"Failed to load stego image: {stego_path}")

            # 确保图像尺寸正确 (必要时调整大小)
            if cover_data.shape != (256, 256):
                cover_data = cv2.resize(cover_data, (256, 256))
            if stego_data.shape != (256, 256):
                stego_data = cv2.resize(stego_data, (256, 256))

            # 堆叠成 (2, H, W) 形状
            data = np.stack([cover_data, stego_data])

            label = np.array([0, 1], dtype='int32')
            sample = {'data': data, 'label': label}

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error loading sample {idx} (file_index: {file_index}): {str(e)}")
            # 返回一个空样本或跳过该样本
            return {'data': np.zeros((2, 256, 256)), 'label': np.array([0, 1])}


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE

    # 获取项目根目录（假设covnet.py在项目根目录下）
    project_root = Path(__file__).parent

    # 配置数据集路径
    BOSSBASE_COVER_DIR = str(project_root / 'data' / 'BOWS2-256')
    BOSSBASE_STEGO_DIR = str(project_root / 'data' / 'BOWS2-suniward')
    BOWS_COVER_DIR = ''  # 如果不需要BOSSBASE数据集则保留空字符串
    BOWS_STEGO_DIR = ''

    # 配置索引文件路径
    TRAIN_INDEX_PATH = str(project_root / 'index_list' / 'bossbase_train_index.npy')
    VALID_INDEX_PATH = str(project_root / 'index_list' / 'bossbase_valid_index.npy')
    TEST_INDEX_PATH = str(project_root / 'index_list' / 'bossbase_test_index.npy')
    # 01 001 0001

    PARAMS_NAME = '{}-{}-{}-{:.2f}-params.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, PROP)
    LOG_NAME = '{}-{}-{}-{:.2f}-model_log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, PROP)

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    setLogger(LOG_PATH, mode='w')

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


    train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                              train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                              eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                             eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    # 加载SRM权重
    try:
        srm_weights = np.load('./SRM_Kernels1.npy')
        logging.info("Successfully loaded SRM filters")
    except Exception as e:
        logging.error(f"Failed to load SRM filters: {str(e)}")
        return
    model = SteganalysisModel(srm_weights).to(device)
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

    # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('Load stego in {}'.format(BOSSBASE_STEGO_DIR))
        logging.info('Load index in {}'.format(TEST_INDEX_PATH))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc = 0.0

    train_hist = {}
    train_hist['acc'] = []
    train_hist['err'] = []

    for epoch in range(startEpoch, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)  # 先训练（optimizer.step() 在 train() 内部调用）
        scheduler.step()  # 再调整学习率

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)

            train_hist['acc'].append(best_acc)
            train_hist['err'].append(1 - best_acc)
            acc_plot(train_hist, OUTPUT_PATH, model_name=STEGANOGRAPHY)

    logging.info('\nTest set accuracy: \n')

    # load best parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)

    test_acc = evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)  #
    print('test_acc', test_acc)
    print('test_err', 1 - test_acc)


def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--DATASET_INDEX',
        help='Path for loading dataset',
        type=str,
        default='1'
    )

    parser.add_argument(
        '-alg',
        '--STEGANOGRAPHY',
        help='embedding_algorithm',
        type=str,
        default='BOWS2-suniward-stego-1'
    )

    parser.add_argument(
        '-rate',
        '--EMBEDDING_RATE',
        help='embedding_rate',
        type=str,
        choices=['0.1', '0.2', '0.3', '0.4'],
        # required=True
        default='0.4'
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3'],
        required=True
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = myParseArgs()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)


