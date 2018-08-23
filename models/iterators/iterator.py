import torch
import torch.nn as nn

import utils

class Iterator(nn.Module):
  def __init__(self, act=None):
    super(Iterator, self).__init__()
    self.act = act
    # Finite difference kernels
    self.fd_update_kernel = torch.Tensor(utils.fd_update_kernel).view(1, 1, 3, 3).cuda()
    self.fd_loss_kernel = torch.Tensor(utils.fd_loss_kernel * (-0.25)).view(1, 1, 3, 3).cuda()

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

  def iter_step(self, x, bc):
    '''
    One step of iteration.
    x: (batch_size, image_size, image_size)
    '''
    return self.forward(x.unsqueeze(1), bc).squeeze(1)

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    raise NotImplementedError


class BasicIterator(Iterator):
  def __init__(self, act):
    super(BasicIterator, self).__init__(act)

    self.layers = nn.Sequential(
                      nn.Conv2d(1, 1, 3, bias=False))
    # Initialize
    initial_weight = torch.Tensor(utils.fd_update_kernel).view(1, 1, 3, 3)
    self.layers[0].weight = nn.Parameter(initial_weight)

  def forward(self, x, bc):
    y = self.layers(x)
    y = self.activation(y)
    # y.size(): batch_size x 1 x (image_size - 2) x (image_size - 2)
    y = y.squeeze(1)
    y = utils.pad_boundary(y, bc)
    y = y.unsqueeze(1) # batch_size x 1 x image_size x image_size
    return y
