import torch
import torch.nn as nn

import utils

class Iterator(nn.Module):
  def __init__(self, act=None):
    super(Iterator, self).__init__()
    self.act = act
    self.n_operations = None
    # Finite difference kernels
    self.fd_update_kernel = utils.update_kernel.view(1, 1, 3, 3).cuda()
    self.fd_loss_kernel = utils.loss_kernel.view(1, 1, 3, 3).cuda()

  def activation(self, x):
    ''' Apply activation '''
    if self.act == 'sigmoid':
      x = torch.sigmoid(x)
    elif self.act == 'clamp':
      x = x.clamp(0, 1)
    elif self.act == 'none':
      pass
    else:
      raise NotImplementedError
    return x

  def iter_step(self, x, bc, f):
    '''
    One step of iteration.
    x, f: (batch_size, image_size, image_size)
    '''
    return self.forward(x, bc, f)

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    raise NotImplementedError

  def name(self):
    raise NotImplementedError
