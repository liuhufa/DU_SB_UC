

import math
import json
import random
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.nn import Parameter
import torch.storage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'    # CPU is even faster :(

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

# 定义激活函数，lambda是匿名函数，相当于简化版def
# Eq. 15 ~ 16 from arXiv:2306.16264
σ = F.sigmoid
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))

# 基本的DU_SB
class DU_SB(nn.Module):

  ''' arXiv:2306.16264 Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''
  # 构造函数，初始化模型的主要参数
  # T是迭代次数
  # batch_size是每批次处理的样本数量

  def __init__(self, T:int, batch_size:int=100):
    super().__init__()

    self.T = T
    self.batch_size = batch_size
    # Eq. 4
    # 在0到1之间生成线性等间隔张量
    self.a = torch.linspace(0, 1, T)

    # the T+2 trainable parameters :)
    # 长度为T的一维张量，初始值为1
    self.Δ = Parameter(torch.ones  ([T],    dtype=torch.float32), requires_grad=True)
    # 标量张量，初始值1
    self.η = Parameter(torch.tensor([1.0],  dtype=torch.float32), requires_grad=True)


  def forward(self, J, h, **kwargs) -> Tensor:
    # 得到J和h
    ''' DU-SB part '''
    # Eq. 6 and 12
    B = self.batch_size
    N = J.shape[0]
    # from essay, this will NOT work
    #c_0: float = 2 * math.sqrt((N - 1) / (J**2).sum())
    # from qaia lib
    # 参考自量子启发式赛道代码
    c_0: Tensor = 0.5 * math.sqrt(N - 1) / torch.linalg.norm(J, ord='fro')

    # rand init x and y 初始化
    x = 0.02 * (torch.rand(N, B, device=J.device) - 0.5)
    y = 0.02 * (torch.rand(N, B, device=J.device) - 0.5)

    # 进行一次SB
    # Eq. 11 ~ 14
    for k, Δ_k in enumerate(self.Δ):
      y = y + Δ_k * (-(1 - self.a[k]) * x + self.η * c_0 * (J @ x + h))
      x = x + Δ_k * y
      x = φ_s(x)
      y = y * (1 - ψ_s(x))

    # [B=100, rb*c*Nt=256]
    spins = x.T

    return spins


def loss_uc(spins:Tensor, x:Tensor, loss_fn:str='mse') -> Tensor:
  ''' differentiable version of compute_ber() '''
  if loss_fn in ['l2', 'mse']:
    return F.mse_loss(spins, x)
  elif loss_fn in ['l1', 'mae']:
    return F.l1_loss(spins, x)



def load_data(limit:int) -> List[Tuple]:
  dataset = []
  for idx in tqdm(range(100)):
    if idx > limit > 0: break
    # 读取最优解文件 (假设文件名为 solution_case_1.csv)
    solution_file = f'data_uc/solution_case_{idx+1}.csv'

    solution_df = pd.read_csv(solution_file)  # 读取整个 CSV 文件
    x = solution_df['Value'].values # 选择 'Value' 列，并展平成一维数组
    x = x * 2 - 1  # 对 x 的每个值进行变换：*2 - 1
    # 读取矩阵文件 (假设文件名为 matrix_case_1.csv)
    matrix_file = f'data_uc/Q_UC_N10_matrix_case{idx+1}.csv'
    total_columns = len(pd.read_csv(matrix_file, nrows=1).columns)
    J = pd.read_csv(matrix_file, usecols=range(1, total_columns)).values

    # 读取单行矩阵文件 (假设文件名为 single_row_case_1.csv)
    single_row_file = f'data_uc/Q_UC_N10_row_case{idx+1}.csv'
    h = pd.read_csv(single_row_file)
    h = h['Value'].values

    J = torch.from_numpy(J).to(device, torch.float32)
    h = torch.from_numpy(h).unsqueeze(1).to(device, torch.float32)
    x = torch.from_numpy(x).to(device, torch.float32)
    dataset.append([J, h, x])
  return dataset


class ValueWindow:

  def __init__(self, nlen=10):
    self.values: List[float] = []
    self.nlen = nlen

  def add(self, v:float):
    self.values.append(v)
    self.values = self.values[-self.nlen:]

  @property
  def mean(self):
    return sum(self.values) / len(self.values) if self.values else 0.0


def train(args):
  print('device:', device)
  print('hparam:', vars(args))
  exp_name = f'{args.M.replace("_", "-")}_T={args.n_iter}_lr={args.lr}{"_overfit" if args.overfit else ""}'

  ''' Data '''
  dataset = load_data(args.limit)

  ''' Model '''
  model: DU_SB = globals()[args.M](args.n_iter, args.batch_size).to(device)
  optim = Adam(model.parameters(), args.lr)

  ''' Ckpt '''
  init_step = 0
  losses = []
  if args.load:
    print(f'>> resume from {args.load}')
    ckpt = torch.load(args.load, map_location='cpu')
    init_step = ckpt['steps']
    losses.extend(ckpt['losses'])
    model.load_state_dict(ckpt['model'], strict=False)
    try:
      optim.load_state_dict(ckpt['optim'])
    except:
      optim_state_ckpt = ckpt['optim']
      optim_state_cur = optim.state_dict()
      optim_state_ckpt['param_groups'][0]['params'] = optim_state_cur['param_groups'][0]['params']
      optim_state_ckpt['state'] = optim_state_cur['state']
      optim.load_state_dict(optim_state_ckpt)

  ''' Bookkeep '''
  loss_wv = ValueWindow(100)
  steps_minor = 0
  steps = init_step

  ''' Train '''
  model.train()
  try:
    pbar = tqdm(total=args.steps-init_step)
    while steps < init_step + args.steps:
      if not args.no_shuffle and steps_minor % len(dataset) == 0:
        random.shuffle(dataset)
      sample = dataset[steps_minor % len(dataset)]

      J, h, x = sample


      spins = model(J, h)

      loss_each = torch.stack([loss_uc(sp, x, args.loss_fn) for sp in spins])
      loss = getattr(loss_each, args.agg_fn)()
      loss_for_backward: Tensor = loss / args.grad_acc
      loss_for_backward.backward()

      loss_wv.add(loss.item())

      steps_minor += 1

      if args.grad_acc == 1 or steps_minor % args.grad_acc:
        optim.step()
        optim.zero_grad()
        steps += 1
        pbar.update()

      if steps % 50 == 0:
        losses.append(loss_wv.mean)
        print(f'>> [step {steps}] loss: {losses[-1]}')
  except KeyboardInterrupt:
    pass

  ''' Ckpt '''
  ckpt = {
    'steps': steps,
    'losses': losses,
    'model': model.state_dict(),
    'optim': optim.state_dict(),
  }
  torch.save(ckpt, LOG_PATH / f'{exp_name}.pth')

  with torch.no_grad():
    params = {
      'deltas': model.Δ.detach().cpu().numpy().tolist(),
      'eta':    model.η.detach().cpu().item(),
    }
    print('params:', params)

    with open(LOG_PATH / f'{exp_name}.json', 'w', encoding='utf-8') as fh:
      json.dump(params, fh, indent=2, ensure_ascii=False)


  plt.plot(losses)
  plt.tight_layout()
  plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=600)


if __name__ == '__main__':
  METHODS = [name for name, value in globals().items() if type(value) == type(DU_SB) and issubclass(value, DU_SB)]

  parser = ArgumentParser()
  parser.add_argument('-M', default='DU_SB', choices=METHODS)
  parser.add_argument('-T', '--n_iter', default=50, type=int)
  parser.add_argument('-B', '--batch_size', default=32, type=int, help='SB candidate batch size')
  parser.add_argument('--steps', default=3000, type=int)
  parser.add_argument('--loss_fn', default='mse', choices=['mse', 'l1', 'bce'])
  parser.add_argument('--agg_fn', default='mean', choices=['mean', 'max'])
  parser.add_argument('--grad_acc', default=1, type=int, help='training batch size')
  parser.add_argument('--lr', default=1e-2, type=float)
  parser.add_argument('--load', help='ckpt to resume')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit dataset n_sample')
  parser.add_argument('--overfit', action='store_true', help='overfit to given dataset')
  parser.add_argument('--no_shuffle', action='store_true', help='no shuffle dataset')
  parser.add_argument('--log_every', default=50, type=int)
  args = parser.parse_args()

  if args.overfit:
    print('[WARN] you are trying to overfit to the given dataset!')

  train(args)
