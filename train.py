import numpy as np
import os

import data
import utils

opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
data_loader = data.get_data_loader(opt)

for epoch in range(opt.start_epoch, opt.n_epochs):
  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    print(bc.size(), final.size(), x.size())
    exit(0)