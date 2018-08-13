import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class UNetIterator(Iterator):
  def __init__(self, act, nf=1):
    super(UNetIterator, self).__init__(act)

    self.fd_update_kernel = torch.Tensor(utils.fd_update_kernel).view(1, 1, 3, 3).cuda()
    self.fd_loss_kernel = torch.Tensor(utils.fd_loss_kernel * (-0.25)).view(1, 1, 3, 3).cuda()

    self.downsample = nn.AvgPool2d(2, stride=2)
    self.layers = nn.Conv2d(1, 1, 3, stride=1, padding=0, bias=False)

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    y = F.conv2d(x, self.fd_loss_kernel)
    y = self.downsample(y)
    # Replicate padding. Conv2d uses zero-pad, which doesn't make sense for us.
    y = F.pad(y, (1, 1, 1, 1), 'replicate')
    y = self.layers(y)
    # Upsample: (image_size - 2, image_size - 2)
    y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)

    y = x[:, :, 1:-1, 1:-1] + y # residual
    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc).unsqueeze(1) # same size as x
    return y
