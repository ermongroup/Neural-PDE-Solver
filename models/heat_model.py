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

    if opt.iterator == 'jacobi':
      self.iterator = JacobiIterator().cuda()
      self.n_operations = 1
      self.is_train = False
    elif opt.iterator == 'multigrid':
      self.iterator = MultigridIterator(opt.multigrid_n_layers, 1, 1).cuda()
      self.n_operations = 4 * (4 / 3)
      self.is_train = False
    elif opt.iterator == 'basic':
      self.iterator = BasicIterator(opt.activation).cuda()
      self.n_operations = 1
    elif opt.iterator == 'unet':
      self.iterator = UNetIterator(opt.activation).cuda()
      self.n_operations = 3
    elif opt.iterator == 'conv':
      self.iterator = ConvIterator(opt.activation, opt.conv_n_layers).cuda()
      self.n_operations = 1 + opt.conv_n_layers
    else:
      raise NotImplementedError
    self.nets['iterator'] = self.iterator

    if self.is_train:
      self.criterion_mse = nn.MSELoss().cuda()
      if opt.optimizer == 'sgd':
        self.optimizer = optim.SGD(self.iterator.parameters(), lr=opt.lr_init)
      elif opt.optimizer == 'adam':
        self.optimizer = optim.Adam(self.iterator.parameters(), lr=opt.lr_init)
      else:
        raise NotImplementedError
      # Hyperparameters
      self.max_iter_steps = opt.max_iter_steps
      self.max_iter_steps_from_gt = opt.max_iter_steps_from_gt
      self.lambdas = {'gt': opt.lambda_gt}

  def train(self, x, gt, bc):
    '''
    x, gt: size (batch_size x image_size x image_size)
    '''
    if not self.is_train:
      return {}

    x = x.cuda()
    gt = gt.cuda()
    bc = bc.cuda()
    loss_dict = {}

    if self.lambdas['gt'] < 1:
      N = np.random.randint(1, self.max_iter_steps + 1)
      # N-1 iterations from x
      y = x.detach()
      for i in range(N - 1):
        y = self.iter_step(y, bc).detach()
      # One more iteration (no detach)
      y = self.iter_step(y, bc)
      loss_x = self.criterion_mse(y, gt)
      loss_dict['loss_x'] = loss_x.item()
    else:
      loss_x = 0

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

  def get_activation(self):
    return self.iterator.act

  def change_activation(self, act):
    ''' Change activation function, used to calculate eigenvalues '''
    self.iterator.act = act

  def evaluate(self, x, gt, bc, n_steps, switch_to_fd=-1):
    '''
    x, gt: size (batch_size x image_size x image_size)
    Run Jacobi and our iterator for n_steps iterations, and calculate errors.
    Return a dictionary of errors: size (batch_size x (n_steps + 1)).
    '''
    bc = bc.cuda()
    x = x.cuda()
    gt = gt.cuda()
    starting_error = utils.l2_error(x, gt).cpu()
    results = {}

    # Jacobi
    fd_errors = [starting_error]
    x_fd = x.detach()
    for i in range(n_steps):
      x_fd = utils.fd_step(x_fd, bc).detach()
      fd_errors.append(utils.l2_error(x_fd, gt).cpu())
    fd_errors = torch.stack(fd_errors, dim=1)
    fd_errors = fd_errors / fd_errors[:, :1] # Normalize by starting_error
    results['Jacobi errors'] = fd_errors

    # error of model
    errors = [starting_error]
    x_model = x.detach()
    for i in range(n_steps):
      x_model = self.iter_step(x_model, bc).detach()
      errors.append(utils.l2_error(x_model, gt).cpu())
    errors = torch.stack(errors, dim=1)
    errors = errors / errors[:, :1] # Normalize by starting_error
    results['model errors'] = errors

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
      errors = torch.stack(errors, dim=1)
      errors = errors / errors[:, :1]
      results['mix errors'] = errors

    return results
