import numpy as np
import os
import random
import torch
import torch.utils.data as data


def make_dataset(root, is_train, max_temp, random_start):
  bc = np.load(os.path.join(root, 'bc.npy'))
  bc /= max_temp
  total_instances = len(bc)
  split = int(0.8 * len(bc)) # 8-2 split
  if is_train:
    bc = bc[:split]
    indices = np.arange(split)
  else:
    bc = bc[split:]
    indices = np.arange(split, total_instances)

  # bc: N x batch_size x 4
  data = []
  for i in indices:
    path = os.path.join(root, 'frames', '{:04d}.npy'.format(i))
    frames = np.load(path) # batch_size x length x image_size x image_size
    if not random_start:
      # Take first and last frame only, (batch_size x 2 x image_size x image_size)
      frames = np.stack([frames[:, 0], frames[:, -1]], axis=1)
    frames /= max_temp # Normalize
    data.append(frames)
  return bc, data

class HeatDataset(data.Dataset):
  def __init__(self, root, is_train, max_temp, random_start):
    self.bc, self.data = make_dataset(root, is_train, max_temp, random_start)

    self.n_instances, self.batch_size, _ = self.bc.shape
    self.is_train = is_train

  def __getitem__(self, idx):
    i = idx // self.batch_size # which instance
    j = idx % self.batch_size # which batch

    bc = self.bc[i, j]
    frames = self.data[i][j] # length x image_size x image_size
    final = frames[-1]
    results = {'bc': torch.Tensor(bc), 'final': torch.Tensor(final)}

    if self.is_train:
      # Randomly choose a frame, except the last frame
      length = frames.shape[0]
      frame_idx = random.randint(0, length - 1)
      x = frames[frame_idx] # image_size x image_size
      results['x'] = torch.Tensor(x)
    else:
      x = frames[0] # first frame
      results['x'] = torch.Tensor(x) # image_size x image_size

    return results

  def __len__(self):
    return self.n_instances * self.batch_size
