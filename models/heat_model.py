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

    # Hyperparameters
    if opt.is_train:
      self.lambdas = {'gt': opt.lambda_gt}

  def train(self, x, gt, bc):
    '''
    x, gt: size (batch_size x image_size x image_size)
    '''
    x = x.cuda()
    gt = gt.cuda()
    bc = bc.cuda()

    # One iteration from x
    y = self.iter_step(x, bc)
    loss_x = self.criterion_mse(y, gt)
    loss_dict = {'loss_x': loss_x.item()}

    # One iteration from gt
    if self.lambdas['gt'] > 0:
      y_gt = self.iter_step(gt, bc)
      loss_gt = self.criterion_mse(y_gt, gt)
      loss_dict['loss_gt'] = loss_gt.item()
    else:
      loss_gt = 0

    loss = (1 - self.lambdas['gt']) * loss_x + self.lambdas['gt'] * loss_gt
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {'loss': loss_dict}

  def iter_step(self, x, bc):
    ''' Perform one iteration step. '''
    y = self.iterator(x.unsqueeze(1))
    y = y.squeeze(1)
    y = utils.pad_boundary(y, bc)
    return y

  def evaluate(self, x, gt, bc, n_steps=200):
    '''
    x, gt: size (batch_size x image_size x image_size)
    Run fd and our iterator for n_steps iterations, and calculate errors.
    Return errors: size (batch_size x (n_steps + 1)).
    '''
    bc = bc.cuda()
    x = x.cuda()
    gt = gt.cuda()
    starting_error = utils.l2_error(x, gt).cpu()

    # fd
    fd_errors = [starting_error]
    x_fd = x.detach()
    for i in range(n_steps):
      x_fd = utils.fd_step(x_fd, bc).detach()
      fd_errors.append(utils.l2_error(x_fd, gt).cpu())
    fd_errors = torch.stack(fd_errors, dim=1)

    # error of model
    errors = [starting_error]
    x_model = x.detach()
    for i in range(n_steps):
      x_model = self.iter_step(x_model, bc).detach()
      errors.append(utils.l2_error(x_model, gt).cpu())
    errors = torch.stack(errors, dim=1)

    print('Kernel:', self.iterator.layers[0].weight.data)
    print('Bias:', self.iterator.layers[0].bias.data)
    return errors, fd_errors