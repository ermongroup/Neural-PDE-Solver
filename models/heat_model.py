import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .iterator import Iterator
import utils

class HeatModel(BaseModel):
  def __init__(self, opt):
    super(HeatModel, self).__init__()
    self.is_train = opt.is_train

    self.iterator = Iterator().cuda()

  def train(self, x, gt, bc):
    x = x.cuda().unsqueeze(1)
    gt = gt.cuda()
    # One iteration
    y = self.iterator(x)
    y = y.squeeze(1)
    # Add boundaries back
    y = utils.pad_boundary(y, bc)
    return
