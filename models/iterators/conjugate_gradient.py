import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterator import Iterator
import utils

class ConjugateGradient(Iterator):
  def __init__(self, n_iters):
    super(ConjugateGradient, self).__init__()
    self.n_iters = n_iters
    self.n_operations = n_iters

  def forward(self, x, bc):
    '''
    x: size (batch_size x 1 x image_size x image_size)
    return: same size
    '''
    batch_size = x.size(0)

    r = - F.conv2d(x, self.fd_loss_kernel)
    p = r.clone()
    rTr = utils.dot_product(r, r)

    for i in range(self.n_iters):
      p_pad = utils.pad_boundary(p.squeeze(1), torch.zeros(1, 4)).unsqueeze(1)
      Ap = F.conv2d(p_pad, self.fd_loss_kernel)
      pAp = utils.dot_product(p, Ap)
      alpha = (rTr / pAp).view(batch_size, 1, 1, 1)

      # Update
      x = x + alpha * p_pad
      r_new = r - alpha * Ap

      rTr_new = utils.dot_product(r_new, r_new)
      beta = (rTr_new / rTr).view(batch_size, 1, 1, 1)
      p_new = r_new + beta * p

      p = p_new
      r = r_new
      rTr = rTr_new

    return x

  def name(self):
    return 'Conjugate Gradient'
