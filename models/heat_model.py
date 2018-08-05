import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_model import BaseModel
from .iterator import Iterator
import utils

class HeatModel(BaseModel):
  def __init__(self, opt):
    super(HeatModel, self).__init__()
    self.is_train = opt.is_train

    self.iterator = Iterator().cuda()
    self.criterion_mse = nn.MSELoss().cuda()
    self.optimizer = optim.Adam(self.iterator.parameters(), lr=opt.lr_init)
    self.nets['iterator'] = self.iterator

  def train(self, x, gt, bc):
    '''
    x, gt: size (batch_size x image_size x image_size)
    '''
    x = x.cuda()
    gt = gt.cuda()
    bc = bc.cuda()
    # One iteration
    y = self.iter_step(x, bc)

    loss = self.criterion_mse(y, gt)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {'loss': loss.item()}

  def iter_step(self, x, bc):
    ''' Perform one iteration step. '''
    y = self.iterator(x.unsqueeze(1))
    y = y.squeeze(1)
    y = utils.pad_boundary(y, bc)
    return y

  def evaluate(self, x, gt, bc, n_steps=200, evaluate_every=1):
    '''
    x: size (image_size x image_size)
    Compare fd iterations and the iterator.
    '''
    bc = bc.cuda().unsqueeze(0)
    x = x.cuda().unsqueeze(0)
    gt = gt.cuda().unsqueeze(0)
    starting_error = utils.l2_error(x, gt)

    # fd
    fd_errors = [starting_error]
    x_fd = x
    for i in range(n_steps):
      x_fd = utils.fd_step(x_fd.detach(), bc)
      fd_errors.append(utils.l2_error(x_fd, gt))

    # error of model
    errors = [starting_error]
    x_model = x
    for i in range(n_steps):
      x_model = self.iter_step(x_model.detach(), bc)
      errors.append(utils.l2_error(x_model, gt))
    print(self.iterator.layers[0].weight)
    print(self.iterator.layers[0].bias)
    return errors, fd_errors
