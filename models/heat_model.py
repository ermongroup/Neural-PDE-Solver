import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_model import BaseModel
from .get_iterator import get_iterator
import utils

class HeatModel(BaseModel):
  def __init__(self, opt):
    super(HeatModel, self).__init__()

    # Iterator
    self.iterator, self.compare_model, self.operations_ratio, self.is_train = get_iterator(opt)
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

  def train(self, x, gt, bc, f):
    '''
    x, gt: size (batch_size x image_size x image_size)
    '''
    if not self.is_train:
      return {}

    x = x.cuda()
    gt = gt.cuda()
    bc = bc.cuda()
    if f is not None:
      f = f.cuda()
    loss_dict = {}

    if self.lambdas['gt'] < 1:
      N = np.random.randint(1, self.max_iter_steps + 1)
      # N-1 iterations from x
      y = x.detach()
      for i in range(N - 1):
        y = self.iter_step(y, bc, f).detach()
      # One more iteration (no detach)
      y = self.iter_step(y, bc, f)
      loss_x = self.criterion_mse(y, gt)
      loss_dict['loss_x'] = loss_x.item()
    else:
      loss_x = 0

    if self.lambdas['gt'] > 0:
      M = np.random.randint(1, self.max_iter_steps_from_gt + 1)
      y_gt = gt.detach()
      for i in range(M - 1):
        y_gt = self.iter_step(y_gt, bc, f).detach()
      # One more iteration
      y_gt = self.iter_step(y_gt, bc, f)
      loss_gt = self.criterion_mse(y_gt, gt)
      loss_dict['loss_gt'] = loss_gt.item()
    else:
      loss_gt = 0

    loss = (1 - self.lambdas['gt']) * loss_x + self.lambdas['gt'] * loss_gt
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {'loss': loss_dict}

  def iter_step(self, x, bc, f):
    ''' Perform one iteration step. '''
    return self.iterator.iter_step(x, bc, f)

  def H(self, x, bc, f):
    ''' Perform one iteration of H. '''
    return self.iterator.H(x.unsqueeze(1), bc).squeeze(1)

  def get_activation(self):
    return self.iterator.act

  def change_activation(self, act):
    ''' Change activation function, used to calculate eigenvalues '''
    self.iterator.act = act

  def evaluate(self, x, gt, bc, f, n_steps):
    '''
    x, f, gt: size (batch_size x image_size x image_size)
    Run Jacobi and our iterator for n_steps iterations, and calculate errors.
    Return a dictionary of errors: size (batch_size x (n_steps + 1)).
    '''
    bc = bc.cuda()
    x = x.cuda()
    gt = gt.cuda()
    if f is not None:
      f = f.cuda()

    if utils.is_bc_mask(x, bc):
      print('Initializing with zero')
      x = utils.initialize(x, bc, 'zero')
    # Calculate starting error
    starting_error = utils.l2_error(x, gt).cpu()
    results = {}

    if self.iterator.name().startswith('UNet'):
      # Unet, set threshold to be higher
      threshold = 0.01
    else:
      threshold = 0.002

    if self.compare_model is not None:
      # Jacobi
      fd_errors, _ = utils.calculate_errors(x, bc, f, gt, self.compare_model.iter_step,
                                            n_steps, starting_error, threshold)
      results['Jacobi errors'] = fd_errors

    # error of model
    errors, x = utils.calculate_errors(x, bc, f, gt, self.iter_step,
                                       n_steps, starting_error, threshold)
    results['model errors'] = errors

    return results, x
