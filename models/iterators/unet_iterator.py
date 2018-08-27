import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class UNetIterator(Iterator):
  def __init__(self, act, n_layers, pre_smoothing, post_smoothing):
    super(UNetIterator, self).__init__(act)
    self.n_layers = n_layers
    self.n_operations = (pre_smoothing + post_smoothing + 2) * 4 / 3

    # Downsampling, first half of U-Net
    downsampling_layers = []
    for i in range(n_layers):
      layers = []
      if i > 0:
        # Subsample
        layers.append(nn.Conv2d(1, 1, 3, stride=2, padding=0, bias=False))
      for j in range(pre_smoothing):
        layers.append(nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False))
      layers = nn.Sequential(*layers)
      downsampling_layers.append(layers)
    self.downsampling_layers = nn.Sequential(*downsampling_layers)

    # Upsampling
    upsampling_layers = []
    for i in range(n_layers):
      layers = []
      for j in range(post_smoothing):
        layers.append(nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False))
      layers = nn.Sequential(*layers)
      upsampling_layers.append(layers)
    self.upsampling_layers = nn.ModuleList(upsampling_layers)

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    y = F.conv2d(x, self.fd_update_kernel)
    z = x[:, :, 1:-1, 1:-1] - y

    # downsampling
    intermediates = []
    for i in range(self.n_layers):
      z = self.downsampling_layers[i](z)
      if i < self.n_layers - 1:
        intermediates.append(z)

    # Upsampling
    for i in range(self.n_layers):
      if i > 0:
        # Bilinear upsample
        z = F.pad(z, (1, 1, 1, 1))
        new_size = z.size(-1) * 2 - 1
        z = F.interpolate(z, size=new_size, mode='bilinear', align_corners=True)
        z = z[:, :, 1:-1, 1:-1]
      z = self.upsampling_layers[i](z)
      if i > 0:
        # residual
        z = z + intermediates[self.n_layers - i - 1]

    y = y + z
    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc).unsqueeze(1) # same size as x
    return y

  def name(self):
    return 'UNet{}'.format(self.n_layers)
