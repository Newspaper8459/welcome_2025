import os
from pathlib import Path
import random
from typing import Union

import numpy as np

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from continuum.datasets import CIFAR10
from continuum.scenarios import ClassIncremental
from tqdm import tqdm



ABS = Path(__file__).resolve().parents[1]
MODEL_PATH = ABS / 'models' / 'cifar-10_2_2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
]

def seed_everything(seed=0):
  """Fix all random seeds"""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

class CLIPIncrement(nn.Module):
  def __init__(self):
    super().__init__()

    self.model, self.preprocess = clip.load('ViT-B/16', device='cuda')
    self.clip_type = self.model.dtype
    self.logit_scale = self.model.logit_scale

    self.current_class_names = []
    self.adapter = nn.Linear(512, 512, device='cuda')

  def forward(self, image):
    image = image.type(torch.float16)
    image_features = self.encode_image(image)
    image_features = self.adapter(image_features.type(torch.float32)).type(torch.float16)
    text_features = self.encode_text(self.text)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)

    return logits_per_image

  def encode_text(self, text):
    with torch.no_grad():
      text_features = self.model.encode_text(text)

    return text_features

  def encode_image(self, image):
    with torch.no_grad():
      image_features = self.model.encode_image(image)
    return image_features

_, preprocess = clip.load('ViT-B/16', device=device)

dataset_valid = CIFAR10(data_path=ABS / 'input', train=False, download=True)
scenario_valid = ClassIncremental(dataset_valid, increment=2, initial_increment=2, transformations=preprocess.transforms, class_order=list(range(10)))


def per_class_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
  assert y_pred.shape == y_true.shape

  accuracy_per_class = [None]*10

  for cls in list(range(10)):
    mask = (y_true == cls)
    total = mask.sum().item()
    correct = (y_pred[mask] == y_true[mask]).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    accuracy_per_class[int(cls)] = accuracy

  return accuracy_per_class

def run_inference(task_id: int, image: Union[np.ndarray|None]=None) -> list[float]:
  model = CLIPIncrement()
  state_dict = torch.load(MODEL_PATH / f'model_{task_id}.pt')
  model.load_state_dict(state_dict)
  model.current_class_names = classes[:2*(task_id+1)]
  model.text = clip.tokenize([f'a good photo of a {c}' for c in model.current_class_names]).to('cuda')

  if image is None:
    model.eval()
    valid_loader = DataLoader(scenario_valid[:task_id+1], batch_size=32)
    Y = []
    PROBS = []
    for X, y, task_ids in tqdm(valid_loader):
      X = X.to(device)
      Y.append(y)
      with torch.no_grad():
        logits_per_image = model(X)
        probs = logits_per_image.cpu().argmax(axis=1)
      PROBS.append(probs)

    Y = torch.cat(Y).cpu()
    PROBS = torch.cat(PROBS).cpu()
    acc = per_class_accuracy(PROBS, Y)
  else:
    with torch.no_grad():
      print('aaaaa')
      X = model.preprocess(image).to(device)
      logits_per_image = model(X.unsqueeze(0)).squeeze()
      probs = F.softmax(logits_per_image, dim=0)
    acc = probs.cpu().tolist()
    acc.extend([0]*(10-len(acc)))
  print(acc)

  print('Inference successfully done')

  return acc
