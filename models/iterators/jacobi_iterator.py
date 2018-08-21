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

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    y = utils.multigrid(x.squeeze(1), bc, self.n_layers,
                        self.pre_smoothing, self.post_smoothing)
    return y.unsqueeze(1)
