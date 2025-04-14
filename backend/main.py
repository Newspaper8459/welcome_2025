import logging
import json
import os
from pathlib import Path
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms

from tqdm import tqdm

from continuum.datasets import CIFAR10, ImageFolderDataset
from continuum.scenarios import ClassIncremental
from continuum.metrics import Logger


ABS = Path(__file__).resolve().parent

class Config:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  dataset = 'cifar-10'
  increment = 2
  initial_increment = 2
  log_path = ABS / 'log' / f'{dataset}_{initial_increment}_{increment}'
  batch_size_train = 128
  batch_size_valid = 128

  num_epochs = 15

cfg = Config()

def seed_everything(seed=0):
  """Fix all random seeds"""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  os.environ['PYTHONHASHSEED'] = str(seed)

class IncrementalResNet18(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.backbone = models.resnet18()
    # self.transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms

    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 0)
    self.num_classes = 0

  def adaptation(self, increment: int) -> None:
    old_fc = self.backbone.fc
    in_features = old_fc.in_features

    new_fc = nn.Linear(in_features, self.num_classes + increment).to(cfg.device)

    with torch.no_grad():
      new_fc.weight[:-increment] = old_fc.weight.detach().clone()

    self.backbone.fc = new_fc
    self.num_classes += increment

  def forward(self, x) -> None:
    x = self.backbone(x)
    return x

class IncrementalResNet6(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    

    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 0)
    self.num_classes = 0

  def adaptation(self, increment: int) -> None:
    old_fc = self.backbone.fc
    in_features = old_fc.in_features

    new_fc = nn.Linear(in_features, self.num_classes + increment).to(cfg.device)

    with torch.no_grad():
      new_fc.weight[:-increment] = old_fc.weight.detach().clone()

    self.backbone.fc = new_fc
    self.num_classes += increment

  def forward(self, x) -> None:
    x = self.backbone(x)
    return x

model = IncrementalResNet18().to(cfg.device)

dataset_train = CIFAR10(data_path='input', train=True, download=True)
dataset_valid = CIFAR10(data_path='input', train=False, download=True)

preprocess = transforms.Compose([
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

scenario_train = ClassIncremental(dataset_train, increment=2, initial_increment=2, transformations=preprocess.transforms)
scenario_valid = ClassIncremental(dataset_valid, increment=2, initial_increment=2, transformations=preprocess.transforms)

cfg.log_path.mkdir(parents=True, exist_ok=True)
with open(cfg.log_path / 'metrics.json', 'w') as f:
  pass

metric_logger = Logger(list_subsets=['test'])

for task_id in range(len(scenario_valid)):
  logging.info(f'Train for task {task_id} has started.')
  model.adaptation(cfg.initial_increment if task_id == 0 else cfg.increment)

  dataloader_train = DataLoader(scenario_train[task_id], batch_size=cfg.batch_size_train, shuffle=True)
  dataloader_valid = DataLoader(scenario_valid[:task_id+1], batch_size=cfg.batch_size_valid)

  model.train()

  optimizer = optim.AdamW(params=model.parameters())
  scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.num_epochs)

  for i_epoch in range(cfg.num_epochs):
    optimizer.zero_grad()

    loss = torch.tensor(0.0).to(cfg.device)
    tqdm_loader = tqdm(dataloader_train)

    for X, y, task_ids in tqdm_loader:
      X, y = X.to(cfg.device), y.to(cfg.device)

      y_pred = model(X)

      loss = F.cross_entropy(y_pred, y)
      loss.backward()
      optimizer.step()
      scheduler.step()

      tqdm_loader.set_description(f'Epoch: {i_epoch+1}/{cfg.num_epochs} | Loss: {loss.item():.4f}')

  tqdm_loader = tqdm(dataloader_valid)
  model.eval()
  for X, y, task_ids in tqdm_loader:
    X = X.to(cfg.device)

    y_pred = model(X)
    y_pred = F.softmax(y_pred, dim=0)
    metric_logger.add([y_pred.cpu().argmax(dim=1), y, task_ids], subset='test')

  with open(cfg.log_path / 'metrics.json', 'r') as f:
    if task_id == 0:
      d = {}
    else:
      d = json.load(f)
  with open(cfg.log_path / 'metrics.json', 'w') as f:
    d[f'task_{task_id}'] = {
      'task': task_id,
      'acc': round(100 * metric_logger.accuracy, 2),
      'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
      'forgetting': round(100 * metric_logger.forgetting, 6),
      'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
      'bwt': round(100 * metric_logger.backward_transfer, 2),
      'fwt': round(100 * metric_logger.forward_transfer, 2),
    }

    json.dump(d, f, indent=2)
