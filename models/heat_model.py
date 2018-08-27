import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_model import BaseModel
from .iterators import *
import utils

class HeatModel(BaseModel):
  def __init__(self, opt):
    super(HeatModel, self).__init__()
    self.is_train = opt.is_train

    # Iterator
    if opt.iterator == 'jacobi':
      self.iterator = JacobiIterator().cuda()
      self.is_train = False
    elif opt.iterator == 'multigrid':
      self.iterator = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                        opt.mg_post_smoothing).cuda()
      self.is_train = False
    elif opt.iterator == 'cg':
      self.iterator = ConjugateGradient(opt.cg_n_iters)
      self.is_train = False
    elif opt.iterator == 'basic':
      self.iterator = BasicIterator(opt.activation).cuda()
    elif opt.iterator == 'conv':
      self.iterator = ConvIterator(opt.activation, opt.conv_n_layers).cuda()
    elif opt.iterator == 'unet':
      self.iterator = UNetIterator(opt.activation, opt.mg_n_layers,
                                   opt.mg_pre_smoothing, opt.mg_post_smoothing).cuda()
    else:
      raise NotImplementedError
    self.nets['iterator'] = self.iterator

    # Compare to Jacobi methods
    if opt.iterator == 'conv' or opt.iterator == 'basic' or \
       opt.iterator == 'multigrid':
      self.compare_model = JacobiIterator()
    elif opt.iterator == 'unet' or opt.iterator == 'cg':
      self.compare_model = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                             opt.mg_post_smoothing)
    else:
      self.compare_model = None
    # ratio of operations
    if self.compare_model is not None:
      self.operations_ratio = self.iterator.n_operations / self.compare_model.n_operations
    else:
      self.operations_ratio = 1

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
    return self.iterator.iter_step(x, bc)

  def get_activation(self):
    return self.iterator.act

  def change_activation(self, act):
    ''' Change activation function, used to calculate eigenvalues '''
    self.iterator.act = act

  def evaluate(self, x, gt, bc, n_steps):
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

    if self.compare_model is not None:
      # Jacobi
      fd_errors, _ = utils.calculate_errors(x, bc, gt, self.compare_model.iter_step,
                                            n_steps, starting_error)
      results['Jacobi errors'] = fd_errors

    # error of model
    errors, _ = utils.calculate_errors(x, bc, gt, self.iter_step, n_steps, starting_error)
    results['model errors'] = errors

    return results
