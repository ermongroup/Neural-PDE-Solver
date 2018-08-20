import numpy as np
import os
import torch
import torch.nn.functional as F

import utils

def multigrid(x, gt, bc):
  y = utils.fd_step(x, bc)
  residual = y - x
  e = gt - x
  # Ae = -r
  A = utils.loss_kernel.view(1, 1, 3, 3) * (-0.25)
  Ae = F.conv2d(e.unsqueeze(1), A).squeeze(1)
  # z should be all zeros
  z = Ae + residual[:, 1:-1, 1:-1]
  print(z)

  # Solve Ae = -r iteratively
  e = residual
  f = -residual
  for i in range(400):
    e = utils.fd_step(e, torch.zeros(1, 4)) - f

  Ae = F.conv2d(e.unsqueeze(1), A).squeeze(1)
  # z should be all zeros
  z = Ae + residual[:, 1:-1, 1:-1]
  #print(z)

  # e = Se
  e = utils.fd_step(e, torch.zeros(1, 4))
  final = y + e
  print(torch.abs(gt - final))

if __name__ == '__main__':
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/16x16')
  bc = np.load(os.path.join(data_dir, 'bc.npy'))[0][0:1]
  bc = torch.Tensor(bc)
  frames = np.load(os.path.join(data_dir, 'frames', '0000.npy'))[0]
  x = torch.Tensor(frames[0:1]).cuda()
  gt = torch.Tensor(frames[-1:]).cuda()
  multigrid(x, gt, bc)
