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
  def __init__(self):
    super(MultigridIterator, self).__init__()

  def forward(self, x, bc):
    pass
