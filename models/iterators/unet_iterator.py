import torch
from .iterator import Iterator

class UNetIterator(Iterator):
  def __init__(self, activation, nf):
    super(Iterator, self).__init__()

    self.layers = nn.Sequential(
                      nn.Conv2d(1, nf, 4, 2, 0),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.ConvTranspose2d()
                  )
