import torch
import torch.nn as nn

from .iterator import Iterator
import utils

class JacobiIterator(Iterator):
  def __init__(self):
    super(JacobiIterator, self).__init__()
    self.n_operations = 1

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    return utils.fd_step(x, bc, f)

  def name(self):
    return 'Jacobi'


class MultigridIterator(Iterator):
  '''
  Multigrid.
  '''
  def __init__(self, n_layers, pre_smoothing, post_smoothing):
    super(MultigridIterator, self).__init__()
    self.n_layers = n_layers
    self.pre_smoothing = pre_smoothing
    self.post_smoothing = post_smoothing
    self.n_operations = (pre_smoothing + post_smoothing + 2) * 4 / 3

  def multigrid_step(self, x, bc, f, step):
    '''
    One layer of multigrid. Recursive function.
    Find solution x to Ax + b = 0.
    '''
    batch_size, image_size, _ = x.size()
    # Pre smoothing
    for i in range(self.pre_smoothing):
      x = utils.fd_step(x, bc, f)

    if step > 1:
      # Downsample
      if f is not None:
        f_sub = 4 * utils.subsample(f)
      else:
        f_sub = None

      if self.is_bc_mask:
        # Subsample geometry
        bc_sub = utils.subsample(bc.view(batch_size * 2, image_size, image_size))
        bc_sub = bc_sub.view(batch_size, 2, *bc_sub.size()[-2:])
      else:
        bc_sub = bc

      x_sub = utils.restriction(x, bc_sub)
      # Refine x_sub recursively
      x_sub = self.multigrid_step(x_sub, bc_sub, f_sub, step - 1)
      # Upsample
      x = utils.interpolation(x_sub, bc)

    # Post smoothing
    for i in range(self.post_smoothing):
      x = utils.fd_step(x, bc, f)
    return x

  def forward(self, x, bc, f):
    '''
    x, f: size (batch_size x image_size x image_size)
    return: same size
    '''
    y = self.multigrid_step(x, bc, f, self.n_layers)
    return y

  def name(self):
    return 'Multigrid'


class MultigridResidualIterator(Iterator):
  '''
  Multigrid method with residual estimation.
  Doesn't work well in heat transfer since the residual isn't smooth near boundaries.
  '''
  def __init__(self, n_layers, pre_smoothing, post_smoothing):
    super(MultigridResidualIterator, self).__init__()
    self.n_layers = n_layers
    self.pre_smoothing = pre_smoothing
    self.post_smoothing = post_smoothing
    self.n_operations = (pre_smoothing + post_smoothing + 2) * 4 / 3

  def multigrid_step(self, x, bc, f, step):
    '''
    One layer of multigrid. Recursive function.
    Find solution x to Ax + b = f.
    Algorithm:
      - Update rule: u^{k+1} = S u^{k} + b - f
      - Residual r^{k} = u^{k+1} - u^{k} = A u^{k} + b - f
      - Solve A e^{k} = - r^{k} recursively.
      - u' = u^{k} + e^{k}
    '''
    if step == 0:
      return None

    # Pre smoothing
    x = utils.set_boundary(x, bc)
    for i in range(self.pre_smoothing):
      x = utils.fd_step(x, bc, f)
    # Calculate residual
    y = utils.fd_step(x, bc, f)
    r = y - x

    # Solve e: A e = -r
    # Restriction: downsample by 2
    zeros_bc = torch.zeros(1, 4)
    r_sub = utils.restriction(r, zeros_bc)

    # Recursive
    ek_sub = self.multigrid_step(r_sub, zeros_bc, - r_sub, step - 1)

    # Upsample
    if ek_sub is not None:
      ek = utils.interpolation(ek_sub, zeros_bc)
      # Add to x
      x = x + ek

    # Post smoothing
    x = utils.set_boundary(x, bc)
    for i in range(self.post_smoothing):
      x = utils.fd_step(x, bc, f)
    return x

  def forward(self, x, bc, f):
    '''
    x: size (batch_size x image_size x image_size)
    return: same size
    '''
    y = self.multigrid_step(x, bc, f, self.n_layers)
    return y

  def name(self):
    return 'Multigrid residual'
