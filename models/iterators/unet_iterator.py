import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator

class UNetIterator(Iterator):
  def __init__(self, act, nf=4):
    super(UNetIterator, self).__init__(act)

    self.layers = nn.Sequential(
                      nn.Conv2d(1, nf, 4, 2, 0, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.ConvTranspose2d(nf, 1, 4, 2, 1, bias=False)
                  )

  def forward(self, x, bc):
    # Same padding. Conv2d uses zero-pad, which doesn't make sense.
    x = F.pad(x, (1, 1, 1, 1), 'replicate')
    y = self.layers(x)
    y = self.activation(y)
    return y
