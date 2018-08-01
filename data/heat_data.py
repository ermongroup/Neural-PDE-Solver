import numpy as np
import random
import torch
import torch.utils.data as data


def make_dataset(root, is_train):
  bc = np.load(os.path.join(root, 'bc.npy'))
  n_instances = len(bc)
  split = int(0.8 * len(bc))
  if is_train:
    bc = bc[:split]
  else:
    bc = bc[split:]

  # bc: N x batch_size x 4
  data = []
  for i in range(len(bc)):
    path = os.path.join(root, 'frames', '{:04d}.npy'.format(i))
    frames = np.load(path) # batch_size x length x image_size x image_size
    data.append(frames)
  return bc, data

class HeatDataset(data.Dataset):
  def __init__(self, root, is_train, max_temp):
    self.bc, self.data = make_dataset(root, is_train)

    self.n_instances, self.batch_size, _ = self.bc.shape
    self.is_train = is_train
    self.max_temp = max_temp

  def __getitem__(self, idx):
    i = idx // self.batch_size
    j = idx % self.batch_size

    bc = self.bc[i, j]
    frames = self.data[i][j]
    final = frames[-1]
    print(final.shape)
    length = frames.shape[0]
    print(length)
    results = {'bc': bc, 'final': final}

    if self.is_train:
      # Randomly choose a frame
      frame_idx = random.randint(0, length - 1)
      x = frames[frame_idx] / self.max_temp
      print(x.shape)
      results['x'] = x

    return results

  def __len__(self):
    return self.n_instances * self.batch_size
