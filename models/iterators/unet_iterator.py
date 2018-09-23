import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class UNetIterator(Iterator):
  def __init__(self, act, n_layers, pre_smoothing, post_smoothing):
    super(UNetIterator, self).__init__(act)
    self.n_layers = n_layers
    self.pre_smoothing = pre_smoothing
    self.post_smoothing = post_smoothing
    self.n_operations = (pre_smoothing + post_smoothing + 2) * 4 / 3

    # First half of U-Net
    first_layers = []
    for i in range(n_layers):
      for j in range(pre_smoothing):
        first_layers.append(nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False))
    self.first_layers = nn.ModuleList(first_layers)

    # Pooling layers
    pooling_layers = []
    for i in range(n_layers - 1):
      pooling_layers.append(nn.Conv2d(1, 1, 3, stride=2, padding=0, bias=False))
    self.pooling_layers = nn.ModuleList(pooling_layers)

    # Second half of U-Net
    second_layers = []
    for i in range(n_layers):
      for j in range(post_smoothing):
        second_layers.append(nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False))
    self.second_layers = nn.ModuleList(second_layers)

  def H(self, r, bc):
    '''
    Return H(r).
    '''
    # Get masks first
    if self.is_bc_mask:
      bc_mask = bc[:, 1:, :, :]
      masks = [1 - bc_mask[:, :, 1:-1, 1:-1]]
      for i in range(self.n_layers - 1):
        bc_mask = utils.subsample(bc_mask.squeeze(1)).unsqueeze(1)
        masks.append(1 - bc_mask[:, :, 1:-1, 1:-1])
      # Multiply by mask
      r = r * masks[0]

    intermediates = [] # used for skip connections

    # First half
    for i in range(self.n_layers):
      for j in range(self.pre_smoothing):
        idx = i * self.pre_smoothing + j
        r = self.first_layers[idx](r)
        if self.is_bc_mask:
          r = r * masks[i]
      # Add to intermediates
      intermediates.append(r)
      # Subsample
      if i < self.n_layers - 1:
        r = self.pooling_layers[i](r)
        if self.is_bc_mask:
          r = r * masks[i + 1]

    # Second half
    for i in range(self.n_layers):
      for j in range(self.post_smoothing):
        idx = i * self.post_smoothing + j
        r = self.second_layers[idx](r)
        if self.is_bc_mask:
          r = r * masks[self.n_layers - i - 1]
      # Add skip connections
      r = r + intermediates[self.n_layers - i - 1]
      # Upsample
      if i < self.n_layers - 1:
        r = F.pad(r, (1, 1, 1, 1))
        new_size = r.size(-1) * 2 - 1
        r = F.interpolate(r, size=new_size, mode='bilinear', align_corners=True)
        r = r[:, :, 1:-1, 1:-1]
        if self.is_bc_mask:
          r = r * masks[self.n_layers - i - 2]

    return r

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    x = x.unsqueeze(1)
    y = F.conv2d(x, self.fd_update_kernel)
    if f is not None:
      y = y - f[..., 1:-1, 1:-1]

    r = y - x[:, :, 1:-1, 1:-1]
    r = self.H(r, bc)

    y = y + r
    y = self.activation(y)
    # Set boundary
    y = utils.pad_boundary(y.squeeze(1), bc)
    return y

  def name(self):
    return 'UNet{}'.format(self.n_layers)
