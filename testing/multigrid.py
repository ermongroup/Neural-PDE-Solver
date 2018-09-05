import numpy as np
import os
import torch
import torch.nn.functional as F

import utils

def test_residual(x, gt, bc):
  y = utils.fd_step(x, bc, None)
  residual = y - x
  e = gt - x
  # Ae = -r
  A = utils.loss_kernel.view(1, 1, 3, 3)
  Ae = F.conv2d(e.unsqueeze(1), A).squeeze(1)
  # z should be all zeros
  z = Ae + residual[:, 1:-1, 1:-1]
  print(z)

  # Solve Ae = -r iteratively
  e = residual
  f = -residual
  for i in range(400):
    e = utils.fd_step(e, torch.zeros(1, 4), f)

  Ae = F.conv2d(e.unsqueeze(1), A).squeeze(1)
  # z should be all zeros
  z = Ae + residual[:, 1:-1, 1:-1]
  #print(z)

  # e = Se
  e = utils.fd_step(e, torch.zeros(1, 4), None)
  final = y + e
  print(torch.abs(gt - final))


def test_multigrid(x, gt, bc):
  from models.iterators import MultigridIterator
  print('Multigrid')
  m = MultigridIterator(3, 4, 4)
  y = m(x, bc, None)

def test_subsampling_poisson(x, gt, bc, f):
  print('Subsampling multigrid')
  for i in range(2000):
    x = utils.fd_step(x, bc, f)

  A = utils.loss_kernel.view(1, 1, 3, 3)
  r = F.conv2d(x.unsqueeze(1), A).squeeze(1)
  r = utils.pad_boundary(r, torch.zeros(1, 4)) - f
  print(np.abs(r.cpu().numpy()).max())

  # Subsample
  x_sub = x
  f_sub = f
  for i in range(3):
    f_sub = utils.subsample(f)
    x_sub = utils.restriction(x, bc)
    r_sub = F.conv2d(x_sub.unsqueeze(1), A).squeeze(1)
    r_sub = utils.pad_boundary(r_sub, torch.zeros(1, 4)) - f_sub
    print(x_sub.size())
    print(np.abs(r_sub.cpu().numpy()).max())

def test_upsampling_poisson(x, gt, bc, f):
  print('Upsampling multigrid')
  f_sub = utils.subsample(f)
  x_sub = utils.restriction(x, bc)
  for i in range(1000):
    x_sub = utils.fd_step(x_sub, bc, f_sub)

  # Upsample
  x = utils.interpolation(x_sub, bc)

  A = utils.loss_kernel.view(1, 1, 3, 3)
  r = F.conv2d(x.unsqueeze(1), A).squeeze(1)
  r = utils.pad_boundary(r, torch.zeros(1, 4)) - f
  r = r.cpu().numpy()
  print(r.max())

if __name__ == '__main__':
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/square/65x65')
  bc = np.load(os.path.join(data_dir, 'bc.npy'))[0][0:1] / 100
  bc = torch.Tensor(bc).cuda()
  frames = np.load(os.path.join(data_dir, 'frames', '0000.npy'))[0] / 100
  x = torch.Tensor(frames[0:1]).cuda()
  gt = torch.Tensor(frames[-1:]).cuda()

  test_residual(x, gt, bc)
  test_multigrid(x, gt, bc)

  # Test Poisson
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/square/poisson_65x65')
  bc = np.load(os.path.join(data_dir, 'bc.npy'))[0][0:1] / 100
  bc = torch.Tensor(bc).cuda()
  frames = np.load(os.path.join(data_dir, 'frames', '0000.npy'))[0] / 100
  assert frames.shape == (3, 65, 65), frames.shape
  x = torch.Tensor(frames[0:1]).cuda()
  gt = torch.Tensor(frames[1:2]).cuda()
  f = torch.Tensor(frames[2:3]).cuda()

  test_subsampling_poisson(x, gt, bc, f)
  test_upsampling_poisson(x, gt, bc, f)
