import torch.utils.data as data

from .heat_data import HeatDataset

def get_data_loader(opt):
  if opt.dset_name == 'heat':
    dset = HeatDataset(opt.dset_path, opt.is_train, opt.max_temp)

  else:
    raise NotImplementedError

  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader
