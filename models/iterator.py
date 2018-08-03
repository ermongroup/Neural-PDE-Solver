import torch
import torch.nn as nn

class Iterator(nn.Module):
  def __init__(self):
    super(Iterator, self).__init__()
    layers = [nn.Conv2d(1, 1, 3),
              nn.Sigmoid()]
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    '''
    x: size (batch_size x n_channels x image_size x image_size)
    return: size (batch_size x n_channels x (image_size - 2) x (image_size - 2))
    '''
    batch_size, n_channels, _, _ = x.size()
    y = self.layers(x)
    return y
