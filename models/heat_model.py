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

  def evaluate(self, x_all, bc, evaluate_every=100):
    '''
    x_all: size (n_frames x image_size x image_size)
    '''
    # fd errors
    gt = x_all[-1:, :, :]
    diff = (x_all[:-1] - gt) ** 2
    fd_errors = diff.mean(dim=-1).mean(dim=-1)
    print(fd_errors.numpy())

    # error of model
    bc = bc.cuda().unsqueeze(0)
    x = x_all[0:1, ...].cuda() # 1 x image_size x image_size
    gt = gt.cuda() # 1 x image_size x image_size

    errors = []
    for i in range(1, x_all.size(0) - 1):
      for j in range(evaluate_every):
        x = self.iter_step(x.detach(), bc)
      diff = (x - gt) ** 2
      error = diff.mean(dim=-1).mean(dim=-1)
      errors.append(error.item())
    print(errors)
    print(self.iterator.layers[0].weight)
    print(self.iterator.layers[0].bias)
    return errors
