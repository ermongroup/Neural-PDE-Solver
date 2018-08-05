import copy
import numpy as np
import os

import data
import utils
from models.heat_model import HeatModel

opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
data_loader = data.get_data_loader(opt)
# Validation
val_dataset = data.heat_data.HeatDataset(opt.dset_path, False, opt.max_temp)

model = HeatModel(opt)

for epoch in range(opt.start_epoch, opt.n_epochs):
  model.setup(is_train=True)
  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    model.train(x, final, bc)
    if (step + 1) % opt.log_every == 0:
      print('Epoch {}, step {}'.format(epoch, step))

  if opt.evaluate_every >= 0 and (epoch + 1) % opt.evaluate_every == 0:
    model.setup(is_train=False)
    # Randomly sample test data
    idx = np.random.randint(len(val_dataset))
    data = val_dataset.__getitem__(idx)
    bc, final, x_all = data['bc'], data['final'], data['x_all']
    errors = model.evaluate(x_all, bc)
