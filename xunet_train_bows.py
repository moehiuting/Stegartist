#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV import *  # MPNCOV
from hpf import *
# ==================== 按照论文修改的超参数 ====================
PROP = 0.50
IMAGE_SIZE = 256
BATCH_SIZE = 32  # 论文：64 images = 32 cover/stego pairs
TOTAL_ITERATIONS = 120000  # 论文总迭代次数
INIT_LR = 0.001  # 论文初始学习率
LR_DECAY_ITER = 5000  # 每5000次迭代衰减
LR_DECAY_RATE = 0.9  # 衰减10%

num_levels = 3
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1



OUTPUT_PATH = Path(__file__).stem

def acc_plot(hist, path = '', model_name = ''):
    x = range(len(hist['acc']))
    y1 = hist['acc']
    y2 = hist['err']

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


# Pre-processing Module
# class HPF(nn.Module):
#     def __init__(self):
#         super(HPF, self).__init__()
#
#         # Load 30 SRM Filters
#         all_hpf_list_5x5 = []
#
#         for hpf_item in all_normalized_hpf_list:
#             if hpf_item.shape[0] == 3:
#                 hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
#
#             all_hpf_list_5x5.append(hpf_item)
#
#         hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)
#
#         self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
#         self.hpf.weight = hpf_weight
#
#         # Truncation, threshold = 3
#         self.tlu = TLU(3.0)
#
#     def forward(self, input):
#
#         output = self.hpf(input)
#         output = self.tlu(output)
#
#         return output
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # 修改后的实现
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        # 关键修改：先转换为单一numpy数组
        hpf_array = np.stack(all_hpf_list_5x5)  # shape: (30, 5, 5)
        hpf_weight = nn.Parameter(torch.from_numpy(hpf_array).float().view(30, 1, 5, 5),
                                  requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(3.0)

    def forward(self, input):  # 保持原始参数名input
        # 完全保持原始计算流程
        output = self.hpf(input)  # 第一层卷积
        output = self.tlu(output)  # 激活函数

        return output  # 保持原始返回变量名

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

class AbsWrapper(nn.Module):
    def forward(self, x):
        x = torch.abs(x)
        return x

class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()

        #self.hpf = HPF_kv5()
        self.hpf = HPF_srm6()


        self.group1 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=2, bias = False),
            AbsWrapper(),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2,)
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
            nn.AvgPool2d(kernel_size=256 // 16, stride=256 // 16)
        )

        self.fc1 = nn.Linear(128, 2)

    def forward(self, input):
        output = input

        output = self.hpf(output)

        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output


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


def train(model, device, train_loader, optimizer, scheduler, iteration):
    """修改后的训练函数，基于迭代次数而不是epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for i, sample in enumerate(train_loader):
        # 更新学习率（基于迭代次数）
        if iteration % LR_DECAY_ITER == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= LR_DECAY_RATE
            logging.info(f'Iteration {iteration}: Learning rate decayed to {optimizer.param_groups[0]["lr"]}')

        data_time.update(time.time() - end)
        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], 1, *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        end = time.time()

        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        losses.update(loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iteration += 1

        if iteration % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('Iter: [{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'LR {lr:.6f}'.format(
                iteration, TOTAL_ITERATIONS, batch_time=batch_time,
                data_time=data_time, loss=losses,
                lr=optimizer.param_groups[0]['lr']))

        # 达到总迭代次数时停止
        if iteration >= TOTAL_ITERATIONS:
            break

    return iteration


def initWeights(module):
    """按照论文的权重初始化"""
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            # 论文：Gaussian distribution with σ=0.01
            nn.init.normal_(module.weight.data, mean=0, std=0.01)
            # 禁用卷积层的bias（论文提到bias在BN层中学习）
            if module.bias is not None:
                module.bias.data.zero_()

    elif type(module) == nn.Linear:
        # 最后一层使用Xavier初始化
        nn.init.xavier_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, val=0)

    elif type(module) == nn.BatchNorm2d:
        # BN层初始化
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


def adjust_bn_stats(model, device, train_loader):
  model.train()

  with torch.no_grad():
    for sample in train_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], 1, *shape[2:])  # 添加通道维度
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], 1, *shape[2:])  # 添加通道维度
      label = label.reshape(-1)

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
  logging.info('Eval err: {:.4f}'.format(1-accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)

  return accuracy


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

        # 确保数据是2D灰度图像 [H, W]
        if data.ndim == 3 and data.shape[0] == 2:  # [2, H, W] - cover和stego堆叠
            # 分别对cover和stego进行增强
            cover = data[0]
            stego = data[1]

            rot = random.randint(0, 3)
            cover = np.rot90(cover, rot, axes=[0, 1]).copy()
            stego = np.rot90(stego, rot, axes=[0, 1]).copy()

            if random.random() < 0.5:
                cover = np.flip(cover, axis=1).copy()
                stego = np.flip(stego, axis=1).copy()

            data = np.stack([cover, stego])

        else:
            # 单个图像处理
            rot = random.randint(0, 3)
            data = np.rot90(data, rot, axes=[0, 1]).copy()
            if random.random() < 0.5:
                data = np.flip(data, axis=1).copy()

        return {'data': data, 'label': label}


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # 确保数据是float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # 处理不同的输入形状
        if data.ndim == 3:  # [2, H, W] - cover和stego堆叠
            # 已经是正确的形状，不需要额外处理
            pass
        elif data.ndim == 2:  # [H, W] - 单个图像
            data = np.expand_dims(data, axis=0)  # [1, H, W]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # 归一化到 [0, 1]
        data = data / 255.0

        return {
            'data': torch.from_numpy(data).float(),
            'label': torch.from_numpy(label).long()
        }


class MyDataset(Dataset):
  def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, transform=None):
    self.index_list = np.load(index_path)
    self.transform = transform

    self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
    self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.pgm'

    self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
    self.bows_stego_path = BOWS_STEGO_DIR + '/{}.pgm'

  def __len__(self):
    return self.index_list.shape[0]

  def __getitem__(self, idx):
      file_index = self.index_list[idx]
      cover = cv2.imread(self.bossbase_cover_path.format(file_index), cv2.IMREAD_GRAYSCALE)  # 直接读取灰度图
      stego = cv2.imread(self.bossbase_stego_path.format(file_index), cv2.IMREAD_GRAYSCALE)  # 直接读取灰度图

      if cover is None or stego is None:
          raise ValueError(f"Failed to load image {file_index}")

      # 确保图像尺寸一致
      if cover.shape != stego.shape:
          # 调整到相同尺寸
          cover = cv2.resize(cover, (256, 256))
          stego = cv2.resize(stego, (256, 256))

      # 堆叠cover和stego [2, H, W]
      data = np.stack([cover, stego])

      sample = {'data': data, 'label': np.array([0, 1], dtype='int32')}

      if self.transform:
          sample = self.transform(sample)

      return sample


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

    BOSSBASE_COVER_DIR = './data/BOWS2-256'
    BOSSBASE_STEGO_DIR = f'./data/BOWS2-{STEGANOGRAPHY}'

    BOWS_COVER_DIR = './data/BOWS2-256'
    BOWS_STEGO_DIR = f'./data/BOWS2-{STEGANOGRAPHY}'

    TRAIN_INDEX_PATH = './index_list/bossbase_train_index.npy'
    VALID_INDEX_PATH = './index_list/bossbase_valid_index.npy'
    TEST_INDEX_PATH = './index_list/bossbase_test_index.npy'

    PARAMS_NAME = '{}-{}-{}-{:.2f}-params.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, PROP)
    LOG_NAME = '{}-{}-{}-{:.2f}-model_log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, PROP)

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    setLogger(LOG_PATH, mode='w')

    # 创建数据集
    train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                              train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                              eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,
                             eval_transform)

    # 按照论文设置批量大小：32对 = 64张图像
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

    # 初始化XUNet模型
    model = XuNet().to(device)
    model.apply(initWeights)

    # 按照论文设置优化器：无权重衰减
    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)

    # 总是从迭代0开始训练（移除恢复训练逻辑）
    iteration = 0
    best_acc = 0.0
    train_hist = {'acc': [], 'err': []}

    logging.info('=' * 60)
    logging.info('Starting XUNet training with paper settings:')
    logging.info(f'Algorithm: {STEGANOGRAPHY} at {EMBEDDING_RATE} bpp')
    logging.info(f'Initial LR: {INIT_LR}, Total iterations: {TOTAL_ITERATIONS}')
    logging.info(f'Batch size: 32 pairs (64 images), Momentum: 0.9')
    logging.info(f'LR decay: every {LR_DECAY_ITER} iterations, rate: {LR_DECAY_RATE}')
    logging.info('=' * 60)

    # 基于迭代次数的训练循环
    while iteration < TOTAL_ITERATIONS:
        # 遍历训练集
        for batch_idx, sample in enumerate(train_loader):
            # 更新学习率（基于迭代次数）
            if iteration % LR_DECAY_ITER == 0 and iteration > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= LR_DECAY_RATE
                logging.info(f'Iteration {iteration}: Learning rate decayed to {optimizer.param_groups[0]["lr"]:.6f}')

            # 训练一个batch
            model.train()
            data, label = sample['data'], sample['label']

            # 调整数据形状
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], 1, *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            iteration += 1

            # 打印训练信息
            if iteration % TRAIN_PRINT_FREQUENCY == 0:
                logging.info(f'Iter: [{iteration:6d}/{TOTAL_ITERATIONS}]\t'
                             f'Loss: {loss.item():.4f}\t'
                             f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # 定期验证
            if iteration % 1000 == 0 and iteration > 0:
                adjust_bn_stats(model, device, train_loader)
                current_acc = evaluate(model, device, valid_loader, iteration, optimizer, best_acc, PARAMS_PATH)

                if current_acc > best_acc:
                    best_acc = current_acc
                    # 保存最佳模型
                    all_state = {
                        'original_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'iteration': iteration,
                        'best_accuracy': best_acc
                    }
                    torch.save(all_state, PARAMS_PATH)

                train_hist['acc'].append(current_acc)
                train_hist['err'].append(1 - current_acc)
                acc_plot(train_hist, OUTPUT_PATH, model_name=f'xunet_{STEGANOGRAPHY}')

            # 达到总迭代次数时退出
            if iteration >= TOTAL_ITERATIONS:
                break

        # 内层循环退出检查
        if iteration >= TOTAL_ITERATIONS:
            break

    logging.info('=' * 60)
    logging.info('Training completed!')
    logging.info(f'Reached {TOTAL_ITERATIONS} iterations')
    logging.info('=' * 60)

    # 最终测试
    logging.info('\nFinal test set evaluation:')

    # 加载最佳模型
    if os.path.exists(PARAMS_PATH):
        all_state = torch.load(PARAMS_PATH)
        model.load_state_dict(all_state['original_state'])
        best_acc = all_state.get('best_accuracy', 0)
        logging.info(f'Loaded best model with validation accuracy: {best_acc:.4f}')

    adjust_bn_stats(model, device, train_loader)

    test_acc = evaluate(model, device, test_loader, iteration, optimizer, best_acc, PARAMS_PATH)

    logging.info('=' * 40)
    logging.info('FINAL RESULTS:')
    logging.info(f'Test Accuracy: {test_acc:.4f}')
    logging.info(f'Test Error: {1 - test_acc:.4f}')
    logging.info(f'Best Validation Accuracy: {best_acc:.4f}')
    logging.info('=' * 40)

    # 保存最终结果
    results = {
        'test_accuracy': test_acc,
        'test_error': 1 - test_acc,
        'best_validation_accuracy': best_acc,
        'algorithm': STEGANOGRAPHY,
        'embedding_rate': EMBEDDING_RATE
    }

    results_path = os.path.join(OUTPUT_PATH, f'results_{STEGANOGRAPHY}_{EMBEDDING_RATE}.npy')
    np.save(results_path, results)
    logging.info(f'Results saved to: {results_path}')


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
    default='synch'
  )

  parser.add_argument(
    '-rate',
    '--EMBEDDING_RATE',
    help='embedding_rate',
    type=str,
    choices=['0.1', '0.2', '0.3', '0.4'],
    #required=True
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


