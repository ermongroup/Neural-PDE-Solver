import argparse
import os


class BaseArgs:
  def __init__(self):
    self.is_train, self.split = None, None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n_workers', type=int, default=8, help='number of threads')

    # data
    self.parser.add_argument('--dset_dir', type=str, default=os.path.join(os.environ['HOME'], 'slowbro', 'PDE'))
    self.parser.add_argument('--dset_name', type=str, default='heat')
    self.parser.add_argument('--image_size', type=int, default=64)
    self.parser.add_argument('--batch_size', type=int, default=4)

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str,
                             default=os.path.join(os.environ['HOME'], 'slowbro', 'ckpt'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt_name', type=str, default='test',
                             help='checkpoint name')
    self.parser.add_argument('--load_ckpt_path', type=str, default='',
                             help='checkpoint path to load trained model')
    self.parser.add_argument('--log_every', type=int, default=20, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=10, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1,
                             help='evaluate on val set every x epochs')

    # specific
    self.parser.add_argument('--iterator', type=str, default='basic',
                             choices=['basic', 'unet'],
                             help='specify iterator architecture')
    self.parser.add_argument('--max_temp', type=int, default=100)
    self.parser.add_argument('--n_evaluation_steps', type=int, default=100,
                             help='number of iterations to run when evaluating')
    self.parser.add_argument('--activation', type=str, default='clamp',
                             help='last layer of iterator to make output [0, 1]')

  def parse(self):
    opt = self.parser.parse_args()

    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    image_size_str = '{}x{}'.format(opt.image_size, opt.image_size)
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name, image_size_str)
    if opt.is_train:
      opt.ckpt_name = '{}_{}_iter{}_{}_gt{}_lr{:.0e}_epoch{}'.format(\
                          opt.ckpt_name, opt.iterator, opt.max_iter_steps,
                          opt.max_iter_steps_from_gt, opt.lambda_gt, opt.lr_init, opt.n_epochs)
      opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, image_size_str, opt.ckpt_name)
    else:
      # Note: for testing, dset and ckpt image size might be different.
      opt.ckpt_path = opt.load_ckpt_path

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
