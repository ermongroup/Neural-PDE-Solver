import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class JacobiIterator(nn.Module):
  def __init__(self):
    super(JacobiIterator, self).__init__()
    self.act = None

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    return utils.fd_step(x.squeeze(1), bc).unsqueeze(1)


class MultigridIterator(nn.Module):
  def __init__(self, n_layers, pre_smoothing, post_smoothing):
    super(MultigridIterator, self).__init__()
    self.act = None
    self.n_layers = n_layers
    self.pre_smoothing = pre_smoothing
    self.post_smoothing = post_smoothing

  def multigrid_step(self, x, bc, f, step):
    '''
    One layer of multigrid. Recursive function.
    Find solution x to Ax + b = f.
    Algorithm:
      - Update rule: u^{k+1} = S u^{k} + b - f
      - Residual r^{k} = u^{k+1} - u^{k} = A u^{k} + b - f
      - Solve A e^{k} = - r^{k} recursively.
      - u' = u^{k} + e^{k}
    '''
    if step == 0:
      return None

    # Pre smoothing
    x = utils.set_boundary(x, bc)
    for i in range(self.pre_smoothing):
      x = utils.fd_step(x, bc)
      x = x - f
    # Calculate residual
    y = utils.fd_step(x, bc)
    r = y - x

    # Solve e: A e = -r
    # Restriction: downsample by 2
    zeros_bc = torch.zeros(1, 4)
    r_sub = utils.restriction(r, zeros_bc)

    # Recursive
    ek_sub = self.multigrid_step(r_sub, zeros_bc, - r_sub, step - 1)

    # Upsample
    if ek_sub is not None:
      ek = utils.interpolation(ek_sub, zeros_bc)
      # Add to x
      x = x + ek

    # Post smoothing
    x = utils.set_boundary(x, bc)
    for i in range(self.post_smoothing):
      x = utils.fd_step(x, bc)
      x = x - f
    return x

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    x = x.squeeze(1)
    f = torch.zeros_like(x).cuda()
    y = self.multigrid_step(x, bc, f, self.n_layers)
    return y.unsqueeze(1)
