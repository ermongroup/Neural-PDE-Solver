import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_model import BaseModel
from .iterators import *
import utils

class HeatModel(BaseModel):
  def __init__(self, opt):
    super(HeatModel, self).__init__()
    self.is_train = opt.is_train

    if opt.iterator == 'basic':
      self.iterator = BasicIterator(opt.activation).cuda()
    elif opt.iterator == 'unet':
      self.iterator = UNetIterator(opt.activation).cuda()
    else:
      raise NotImplementedError
    self.nets['iterator'] = self.iterator

    if opt.is_train:
      self.criterion_mse = nn.MSELoss().cuda()
      self.optimizer = optim.Adam(self.iterator.parameters(), lr=opt.lr_init)
      # Hyperparameters
      self.max_iter_steps = opt.max_iter_steps
      self.max_iter_steps_from_gt = opt.max_iter_steps_from_gt
      self.lambdas = {'gt': opt.lambda_gt}

  def train(self, x, gt, bc):
    '''
    x, gt: size (batch_size x image_size x image_size)
    '''
    x = x.cuda()
    gt = gt.cuda()
    bc = bc.cuda()

    N = np.random.randint(1, self.max_iter_steps + 1)
    # N-1 iterations from x
    y = x.detach()
    for i in range(N - 1):
      y = self.iter_step(y, bc).detach()
    # One more iteration (no detach)
    y = self.iter_step(y, bc)
    loss_x = self.criterion_mse(y, gt)
    loss_dict = {'loss_x': loss_x.item()}

    if self.lambdas['gt'] > 0:
      M = np.random.randint(1, self.max_iter_steps_from_gt + 1)
      y_gt = gt.detach()
      for i in range(M - 1):
        y_gt = self.iter_step(y_gt, bc).detach()
      # One more iteration
      y_gt = self.iter_step(y_gt, bc)
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
    y = self.iterator(x.unsqueeze(1), bc)
    y = y.squeeze(1)
    return y

  def evaluate(self, x, gt, bc, n_steps, switch_to_fd=-1):
    '''
    x, gt: size (batch_size x image_size x image_size)
    Run fd and our iterator for n_steps iterations, and calculate errors.
    Return a dictionary of errors: size (batch_size x (n_steps + 1)).
    '''
    bc = bc.cuda()
    x = x.cuda()
    gt = gt.cuda()
    starting_error = utils.l2_error(x, gt).cpu()
    results = {}

    # fd
    fd_errors = [starting_error]
    x_fd = x.detach()
    for i in range(n_steps):
      x_fd = utils.fd_step(x_fd, bc).detach()
      fd_errors.append(utils.l2_error(x_fd, gt).cpu())
    results['fd errors'] = torch.stack(fd_errors, dim=1)

    # error of model
    errors = [starting_error]
    x_model = x.detach()
    for i in range(n_steps):
      x_model = self.iter_step(x_model, bc).detach()
      errors.append(utils.l2_error(x_model, gt).cpu())
    results['model errors'] = torch.stack(errors, dim=1)

    # Run model until switch_to_fd, then switch to fd
    errors = [starting_error]
    x_model = x.detach()
    if switch_to_fd > 0:
      for i in range(switch_to_fd):
        x_model = self.iter_step(x_model, bc).detach()
        errors.append(utils.l2_error(x_model, gt).cpu())
      for j in range(n_steps - switch_to_fd):
        x_model = utils.fd_step(x_model, bc).detach()
        errors.append(utils.l2_error(x_model, gt).cpu())
      results['mix errors'] = torch.stack(errors, dim=1)

    return results
