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
import torch.nn.functional as F
from srm_filter_kernel import all_normalized_hpf_list
# from srm_filter_kernel import all_hpf_list
from MPNCOV.python import MPNCOV



PROP = 0.50
IMAGE_SIZE = 256
BATCH_SIZE = 64 // 2
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


OUTPUT_PATH = Path(__file__).stem
try:
  os.makedirs(OUTPUT_PATH)
except OSError:
  pass
def acc_plot(hist, path = '', model_name = ''):
    x = range(len(hist['acc']))
    y1 = hist['acc'].cpu().numpy() if torch.is_tensor(hist['acc']) else np.array(hist['acc'])
    y2 = hist['err'].cpu().numpy() if torch.is_tensor(hist['err']) else np.array(hist['err'])

    plt.plot(x,y1,label='acc')
    plt.plot(x,y2,label='err')

    plt.xlabel('Iter')
    plt.ylabel('Acc')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'acc_' + model_name + '.png')

    plt.savefig(path)

    plt.close()

# 保持原有HPF和TLU实现不变 -------------------------------------------------
class TLU(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, -self.threshold, self.threshold)


class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
    # for hpf_item in all_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    # 原始代码（效率低）
    #hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

    # 优化后的代码（先转换成 numpy.array，再转 Tensor）
    hpf_weight = nn.Parameter(torch.tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5), requires_grad=False)

    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight

    self.tlu = TLU(3.0)

    # self.sc_bn_1 = nn.BatchNorm2d(30)


    # nn.init.constant_(self.sc_bn.weight, 1.0)


  def forward(self, input):

    output = self.hpf(input)
    output = self.tlu(output)


    return output


# 新增Transformer组件 -----------------------------------------------------
class PositionalEncoding(nn.Module):
    """位置编码层"""

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


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # 保持与原始代码相同的eps
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),  # 对应原始代码的gelu激活
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 与原始代码完全相同的残差结构
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """完整的ViT模块"""

    def __init__(self, in_channels=1, patch_size=4, dim=128, depth=4, num_heads=4, mlp_ratio=4):
        super().__init__()
        # 对应原始代码的projected_patches生成
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = PositionalEncoding(dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # 1. 分块嵌入 [B, C, H, W] -> [B, dim, h, w]
        x = self.patch_embed(x)

        # 2. 展平并添加位置编码 [B, dim, h, w] -> [B, h*w, dim]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.pos_embed(x)

        # 3. Transformer编码
        x = self.blocks(x)
        x = self.norm(x)

        # 4. 全局平均池化 [B, h*w, dim] -> [B, dim]
        return x.mean(dim=1)

    # 修改后的HybridModel -----------------------------------------------------


# === 新增LSNet核心模块 === #
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(
                1 - self.drop).detach()
        else:
            return x + self.m(x)


class LSConv(nn.Module):  # !!! 新增局部空间卷积
    def __init__(self, dim):
        super().__init__()
        self.conv = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.conv1 = Conv2d_BN(dim, dim, 1, 1, 0, groups=dim)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.attn(x) * (self.conv(x) + self.conv1(x))


# === 修改后的HybridModel === #
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 保持原有HPF层不变
        self.hpf = HPF()

        # !!! 修改CNN分支结构
        self.group2 = nn.Sequential(
            Conv2d_BN(30, 32, 3, 1, 1),
            nn.ReLU(),
            LSConv(32),  # 替换原始卷积
            nn.AvgPool2d(3, stride=2, padding=1)
        )

        self.group3 = nn.Sequential(
            Conv2d_BN(32, 64, 3, 1, 1),
            nn.ReLU(),
            Residual(LSConv(64)),  # !!! 增加残差连接
            nn.AvgPool2d(3, stride=2, padding=1)
        )

        self.group4 = nn.Sequential(
            Conv2d_BN(64, 64, 3, 1, 1),
            nn.ReLU(),
            Conv2d_BN(64, 128, 3, 1, 1),
            Residual(LSConv(128)),  # !!! 关键改进点
            nn.AvgPool2d(3, stride=2, padding=1)
        )

        # !!! 完全替换group5为LSNet风格块
        self.group5 = nn.Sequential(
            Residual(LSConv(128)),
            Conv2d_BN(128, 256, 3, 1, 1),
            Residual(LSConv(256))
        )

        # 增强Transformer分支
        self.transformer = VisionTransformer(
            in_channels=1,
            patch_size=16,
            dim=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4
        )

        # !!! 改进的特征融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3个融合权重
            nn.Softmax(dim=1)
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),  # !!! 新增BN层
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # !!! 增加正则化
            nn.Linear(128, 2)
        )

        # MPNCOV操作
        self.covpool = MPNCOV.Covpool.apply
        self.sqrtm = lambda x: MPNCOV.Sqrtm.apply(x, 5)
        self.triuvec = MPNCOV.Triuvec.apply

    def forward(self, raw_img, hpf_img):
        # CNN分支
        x = self.group2(hpf_img)
        x = self.group3(x)
        x = self.group4(x)
        cnn_feat = self.group5(x)

        # Transformer分支
        trans_feat = self.transformer(raw_img)

        # !!! 改进的特征处理
        cnn_vec1 = F.avg_pool2d(cnn_feat, cnn_feat.size()[2:]).flatten(1)  # [B,256]
        cnn_vec2 = self.triuvec(self.sqrtm(self.covpool(cnn_feat))).squeeze(-1)[:, :128]  # [B,128]

        # 动态融合
        gate_weights = self.fusion_gate(torch.cat([cnn_vec1, trans_feat], dim=1))  # [B,3]
        combined = torch.cat([
            gate_weights[:, 0:1] * cnn_vec1,
            gate_weights[:, 1:2] * trans_feat,
            gate_weights[:, 2:3] * cnn_vec2
        ], dim=1)  # [B,512]

        return self.fc(combined)


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

        # 获取双输入数据
        raw_data = sample['raw_data'].to(device)  # [B, 2, 1, H, W]
        hpf_data = sample['hpf_data'].to(device)  # [B, 2, 30, H, W]
        label = sample['label'].to(device).long()  # [B, 2]

        # 调整数据形状
        batch_size = raw_data.shape[0]
        raw_data = raw_data.reshape(-1, *raw_data.shape[2:])  # [B×2, 1, H, W]
        hpf_data = hpf_data.reshape(-1, *hpf_data.shape[2:])  # [B×2, 30, H, W]
        label = label.reshape(-1)  # [B×2]

        optimizer.zero_grad()
        output = model(raw_data, hpf_data)  # 注意模型现在接受两个输入

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), raw_data.size(0))

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
    model.train()  # 确保BN处于训练模式

    with torch.no_grad():
        for sample in train_loader:
            # 获取并展平数据
            hpf_data = sample['hpf_data'].to(device)  # [B,2,30,H,W]
            hpf_data = hpf_data.view(-1, *hpf_data.shape[2:])  # [B*2,30,H,W]

            # 跳过HPF层，直接使用预处理好的特征
            _ = model.group2(hpf_data)
            _ = model.group3(_)
            _ = model.group4(_)
            _ = model.group5(_)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            raw_data = sample['raw_data'].to(device)
            hpf_data = sample['hpf_data'].to(device)
            label = sample['label'].to(device)

            # 处理成对数据
            if len(raw_data.shape) == 5:
                raw_data = raw_data.reshape(-1, *raw_data.shape[2:])
                hpf_data = hpf_data.reshape(-1, *hpf_data.shape[2:])
                label = label.reshape(-1)

            output = model(raw_data, hpf_data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    if accuracy > best_acc and epoch > 10:
        best_acc = accuracy
        torch.save({
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }, PARAMS_PATH)

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
    def __init__(self):
        # 不在这里初始化设备，延迟到第一次调用时处理
        self.hpf_layer = None
        self.device = None

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # 延迟初始化（只在主线程初始化CUDA）
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.hpf_layer = HPF().eval().to(self.device)

        # 原始图像处理
        raw_data = np.expand_dims(data, axis=1).astype(np.float32)  # [2,1,H,W]
        raw_tensor = torch.from_numpy(raw_data)

        # HPF处理
        with torch.no_grad():
            if self.device.type == 'cuda':
                raw_tensor = raw_tensor.to(self.device, non_blocking=True)
            hpf_data = self.hpf_layer(raw_tensor)
            if self.device.type == 'cuda':
                hpf_data = hpf_data.cpu()  # 移回CPU以便多进程处理

        return {
            'raw_data': raw_tensor.cpu(),  # 确保返回CPU上的张量
            'hpf_data': hpf_data.cpu(),  # 确保返回CPU上的张量
            'label': torch.from_numpy(label).long()
        }


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
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # 必须在主程序中设置

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
    BOSSBASE_COVER_DIR = str(project_root / 'data' / 'BossBase-256')
    BOSSBASE_STEGO_DIR = str(project_root / 'data' / 'BossBase-suniward')
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

    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = HybridModel().to(device)
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)


    param_groups = [
        # Transformer分支使用更低学习率(LR*0.1)
        {'params': [p for n, p in model.named_parameters() if 'transformer' in n], 'lr': LR * 0.1,
         'weight_decay': WEIGHT_DECAY},

        # HPF层使用最低学习率(LR*0.01)
        {'params': [p for n, p in model.named_parameters() if 'hpf' in n], 'lr': LR * 0.01, 'weight_decay': 0},
        # 通常HPF不需要weight_decay

        # CNN主干网络(group2-5)使用基准学习率
        {'params': [p for n, p in model.named_parameters() if any(f'group{i}' in n for i in range(2, 6))], 'lr': LR,
         'weight_decay': WEIGHT_DECAY},

        # 分类头(fc)使用稍高学习率(LR*1.2)
        {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': LR * 1.2, 'weight_decay': WEIGHT_DECAY},

        # 其他参数(如BN层)使用基准学习率
        {'params': [p for n, p in model.named_parameters() if not any(
            k in n for k in ['transformer', 'hpf', 'fc'] + [f'group{i}' for i in range(2, 6)]
        )], 'lr': LR, 'weight_decay': WEIGHT_DECAY}
    ]

    optimizer = optim.SGD(param_groups, momentum=0.9)

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

    # 2. 重置best_acc为0（测试阶段独立统计）
    test_best_acc = 0.0
    adjust_bn_stats(model, device, train_loader)

    test_acc = evaluate(model, device, test_loader, epoch, optimizer, test_best_acc, PARAMS_PATH)  #
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
        default='BossBase-suniward-stego-1'
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
