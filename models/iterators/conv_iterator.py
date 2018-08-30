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

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    x = x.unsqueeze(1)
    z = F.conv2d(x, self.fd_update_kernel, padding=0)
    if f is not None:
      z = z - f.unsqueeze(1)[:, :, 1:-1, 1:-1]
    y = z - x[:, :, 1:-1, 1:-1]

    if self.is_bc_mask:
      mask = 1 - bc[:, 1:, 1:-1, 1:-1] # foreground mask

    for i in range(self.n_layers):
      y = self.layers[i](y)
      if self.is_bc_mask:
        y = y * mask

    y = z + y # residual

    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc)
    return y

  def name(self):
    return 'Conv{}'.format(self.n_layers)
