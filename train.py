import copy
import numpy as np
import os
import torch
import torchvision

import data
import utils
from models.heat_model import HeatModel

opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
data_loader = data.get_data_loader(opt)
# Validation
val_opt = copy.deepcopy(opt)
val_opt.is_train = False
val_opt.batch_size = 4
val_loader = data.get_data_loader(val_opt)

model = HeatModel(opt)

for epoch in range(opt.start_epoch, opt.n_epochs):
  model.setup(is_train=True)
  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    loss_dict = model.train(x, final, bc)
    if (step + 1) % opt.log_every == 0:
      print('Epoch {}, step {}'.format(epoch, step))
      vis.add_scalar(loss_dict, epoch * len(data_loader) + step)

  if opt.evaluate_every >= 0 and (epoch + 1) % opt.evaluate_every == 0:
    model.setup(is_train=False)
    # Randomly sample test data
    data = next(iter(val_loader))
    bc, final, x = data['bc'], data['final'], data['x']
    errors, fd_errors = model.evaluate(x, final, bc)
    # Plot error curve
    images = []
    for i in range(errors.size(0)):
      img = utils.plot([{'y': fd_errors[i], 'label': 'fd errors'},
                        {'y': errors[i], 'label': 'model errors'}],
                       {'title': 'iterations', 'ylim': (0, 0.15), 'image_size': (640, 480)})
      img = img.transpose((2, 0, 1)) / 255
      images.append(torch.Tensor(img))
    images = torchvision.utils.make_grid(images, nrow=errors.size(0))
    vis.add_image({'errors': images}, epoch)
