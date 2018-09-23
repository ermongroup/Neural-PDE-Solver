import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class ConvIterator(Iterator):
  def __init__(self, act, n_layers=1):
    super(ConvIterator, self).__init__(act)

    layers = []
    for i in range(n_layers):
      layers += [nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)]
    self.layers = nn.ModuleList(layers)
    self.n_layers = n_layers
    self.n_operations = 1 + n_layers

  def H(self, r, bc):
    '''
    Return H(r).
    '''
    if self.is_bc_mask:
      mask = 1 - bc[:, 1:, 1:-1, 1:-1] # foreground mask
      r = r * mask

    for i in range(self.n_layers):
      r = self.layers[i](r)
      if self.is_bc_mask:
        r = r * mask
    return r

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    x = x.unsqueeze(1)
    y = F.conv2d(x, self.fd_update_kernel, padding=0)
    if f is not None:
      y = y - f.unsqueeze(1)[:, :, 1:-1, 1:-1]

    r = y - x[:, :, 1:-1, 1:-1]
    r = self.H(r)

    y = y + r

    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc)
    return y

  def name(self):
    return 'Conv{}'.format(self.n_layers)
