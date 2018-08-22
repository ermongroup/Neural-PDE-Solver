import torch.utils.data as data

from .heat_data import HeatDataset

def get_dataset(opt):
  if opt.dset_name == 'heat':
    dset = HeatDataset(opt.dset_path, opt.is_train, opt.max_temp, opt.data_limit)
  else:
    raise NotImplementedError
  return dset

def get_data_loader(opt):
  dset = get_dataset(opt)
  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader

def get_random_data_loader(opt):
  '''
  Sample random batch. Used for evaluation during training.
  '''
  dset = get_dataset(opt)
  # Random sampler
  sampler = data.RandomSampler(dset)
  dloader = data.DataLoader(dset, batch_size=opt.batch_size, sampler=sampler)
  return dloader
