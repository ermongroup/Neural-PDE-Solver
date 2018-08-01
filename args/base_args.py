import argparse
import os


class BaseArgs:
  def __init__(self):
    self.is_train, self.split = None, None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
    self.parser.add_argument('--gpus', type=str, default='0', help='visible GPU ids, separated by comma')

    # data
    self.parser.add_argument('--dset_dir', type=str, default=os.path.join(os.environ['HOME'], 'slowbro', 'PDE'))
    self.parser.add_argument('--dset_name', type=str, default='heat')
    self.parser.add_argument('--batch_size', type=int, default=4)

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.environ['HOME'], 'slowbro', 'ckpt'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt_name', type=str, default='test', help='checkpoint name')
    self.parser.add_argument('--log_every', type=int, default=10, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=10, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1, help='evaluate on val set every x epochs')

    # specific
    self.parser.add_argument('--max_temp', type=int, default=100)

  def parse(self):
    opt = self.parser.parse_args()

    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, opt.ckpt_name)

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
