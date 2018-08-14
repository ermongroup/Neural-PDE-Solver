import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class UNetIterator(Iterator):
  def __init__(self, act, nf=1):
    super(UNetIterator, self).__init__(act)

    self.conv1 = nn.Conv2d(1, nf, 3, stride=2, padding=0, bias=False)
    self.conv2 = nn.Conv2d(nf, 1, 3, stride=1, padding=0, bias=False)

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    z = F.conv2d(x, self.fd_update_kernel)
    y = x[:, :, 1:-1, 1:-1] - z

    # replicate padding. Conv2d uses zero-pad, which doesn't make sense for us.
    y = F.pad(y, (0, 1, 0, 1), mode='replicate')
    y = self.conv1(y) # downsample by 2
    y = F.pad(y, (1, 1, 1, 1), mode='replicate')
    y = self.conv2(y)
    # Upsample: (image_size - 2, image_size - 2)
    y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)

    y = z + y # residual

    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc).unsqueeze(1) # same size as x
    return y
