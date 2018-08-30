import numpy as np
import os
import random
import torch
import torch.utils.data as data


def make_dataset(root, is_train, max_temp, data_limit):
  bc = np.load(os.path.join(root, 'bc.npy'))
  bc /= max_temp
  total_instances = len(bc)
  split = int(0.9 * len(bc)) # 9-1 split
  if is_train:
    bc = bc[:split]
    indices = np.arange(split)
  else:
    bc = bc[split:]
    indices = np.arange(split, total_instances)

  if data_limit > 0:
    bc = bc[:data_limit]
    indices = indices[:data_limit]
  # bc: N x batch_size x 4
  data = []
  for i in indices:
    path = os.path.join(root, 'frames', '{:04d}.npy'.format(i))
    frames = np.load(path) # batch_size x length x image_size x image_size
    # Take first and last frame only
    # Size: (batch_size x 2 x image_size x image_size)
    frames = np.stack([frames[:, 0], frames[:, -1]], axis=1)
    frames /= max_temp # Normalize
    data.append(frames)
  return bc, data

class HeatDataset(data.Dataset):
  '''
  Heat Dataset on square geometry.
  '''
  def __init__(self, root, is_train, max_temp, data_limit):
    self.bc, self.data = make_dataset(root, is_train, max_temp, data_limit)

    self.n_instances, self.batch_size, _ = self.bc.shape
    self.is_train = is_train

  def rotate(self, bc, x, final):
    ''' Random rotate '''
    k = random.randint(0, 3)
    if k > 0:
      x = np.rot90(x, k).copy()
      final = np.rot90(final, k).copy()
      if k == 1:
        indices = [3, 2, 0, 1]
      elif k == 2:
        indices = [1, 0, 3, 2]
      elif k == 3:
        indices = [2, 3, 1, 0]
      bc = bc[indices]
    return bc, x, final

  def __getitem__(self, idx):
    i = idx // self.batch_size # which instance
    j = idx % self.batch_size # which batch

    bc = self.bc[i, j]
    frames = self.data[i][j] # length x image_size x image_size
    final = frames[-1]
    x = frames[0] # first frame
    if self.is_train:
      # Random rotate
      bc, x, final = self.rotate(bc, x, final)
    bc = torch.Tensor(bc)
    x = torch.Tensor(x)
    final = torch.Tensor(final)
    bc, x, final = torch.Tensor(bc), torch.Tensor(x), torch.Tensor(final)

    results = {'bc': bc, 'final': final, 'x': x}
    return results

  def __len__(self):
    return self.n_instances * self.batch_size


def make_geometry_dataset(root, is_train, max_temp, data_limit):
  n_instances = len(os.listdir(os.path.join(root, 'frames')))
  # No train test split yet.
  if data_limit > 0 and data_limit < n_instances:
    n_instances = data_limit

  data = []
  for i in range(n_instances):
    path = os.path.join(root, 'frames', '{:04d}.npy'.format(i))
    frames = np.load(path) # batch_size x 4 x image_size x image_size
    frames[:, :3] /= max_temp # Normalize
    data.append(frames)
  return data

class HeatGeometryDataset(data.Dataset):
  '''
  Heat Dataset on square geometry.
  '''
  def __init__(self, root, is_train, max_temp, data_limit):
    self.data = make_geometry_dataset(root, is_train, max_temp, data_limit)

    self.n_instances = len(self.data)
    self.batch_size = self.data[0].shape[0]
    self.is_train = is_train

  def rotate(self, data):
    ''' Random rotate '''
    k = random.randint(0, 3)
    if k > 0:
      data = np.rot90(data, k, axes=(1, 2)).copy()
    return data

  def __getitem__(self, idx):
    i = idx // self.batch_size # which instance
    j = idx % self.batch_size # which batch

    frames = self.data[i][j] # 4 x image_size x image_size
    if self.is_train:
      # Random rotate
      frames = self.rotate(frames)

    x = torch.Tensor(frames[0])
    final = torch.Tensor(frames[1])
    bc = torch.Tensor(frames[2:])

    results = {'x': x, 'final': final, 'bc': bc}
    return results

  def __len__(self):
    return self.n_instances * self.batch_size
