import os
from collections import OrderedDict
import torch


class BaseModel:
  def __init__(self):
    self.nets, self.optimizers, self.schedulers = {}, {}, []

  def setup(self, is_train):
    for _, net in self.nets.items():
      if is_train:
        net.train()
      else:
        net.eval()

  def load(self, ckpt_path, epoch, load_optimizer=False):
    for name, net in self.nets.items():
      path = os.path.join(ckpt_path, 'net_{}_{}.pth'.format(name, epoch))
      if not os.path.exists(path):
        print('{} does not exist, ignore.'.format(path))
        continue
      if torch.cuda.is_available():
        ckpt = torch.load(path)
      else:
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)

      if isinstance(net, torch.nn.DataParallel):
        module = net.module
      else:
        module = net

      try:
        module.load_state_dict(ckpt)
      except:
        print('net_{} and checkpoint have different parameter names'.format(name))
        new_ckpt = OrderedDict()
        for ckpt_key, module_key in zip(ckpt.keys(), module.state_dict().keys()):
          assert ckpt_key.split('.')[-1] == module_key.split('.')[-1]
          new_ckpt[module_key] = ckpt[ckpt_key]
        module.load_state_dict(new_ckpt)

    if load_optimizer:
      for name, optimizer in self.optimizers.items():
        path = os.path.join(ckpt_path, 'optimizer_{}_{}.pth'.format(name, epoch))
        if not os.path.exists(path):
          print('{} does not exist, ignore.'.format(path))
          continue
        ckpt = torch.load(path)
        optimizer.load_state_dict(ckpt)

  def save(self, ckpt_path, epoch):
    for name, net in self.nets.items():
      if isinstance(net, torch.nn.DataParallel):
        module = net.module
      else:
        module = net

      path = os.path.join(ckpt_path, 'net_{}_{}.pth'.format(name, epoch))
      torch.save(module.state_dict(), path)

    for name, optimizer in self.optimizers.items():
      path = os.path.join(ckpt_path, 'optimizer_{}_{}.pth'.format(name, epoch))
      torch.save(optimizer.state_dict(), path)

  def update_lr(self, epoch, n_epochs, lr_init):
    raise NotImplementedError
