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
    self.parser.add_argument('--batch_size', type=int, default=4)
    self.parser.add_argument('--image_size', type=int, default=64)
    # Specific
    self.parser.add_argument('--max_temp', type=int, default=100)
    self.parser.add_argument('--initialization', type=str, default='random',
                             choices=['random', 'zero', 'avg'],
                             help='method to initialize the inner part of the data')
    self.parser.add_argument('--data_limit', type=int, default=-1, help='limit amount of data when testing')

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str,
                             default=os.path.join(os.environ['HOME'], 'slowbro', 'ckpt'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt_name', type=str, default='test',
                             help='checkpoint name')
    self.parser.add_argument('--load_ckpt_path', type=str, default='',
                             help='checkpoint path to load trained model')
    self.parser.add_argument('--log_every', type=int, default=50, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=10, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1,
                             help='evaluate on val set every x epochs')

    # model
    self.parser.add_argument('--iterator', type=str, default='basic',
                             choices=['jacobi', 'basic', 'unet', 'conv', 'multigrid'],
                             help='specify iterator architecture')
    self.parser.add_argument('--n_evaluation_steps', type=int, default=100,
                             help='number of iterations to run when evaluating')
    self.parser.add_argument('--switch_to_fd', type=int, default=-1,
                             help='when to switch to fd, -1 if no switch')
    self.parser.add_argument('--activation', type=str, default='clamp',
                             help='last layer of iterator to make output [0, 1]')
    self.parser.add_argument('--conv_n_layers', type=int, default=1,
                             help='number of layers in the conv iterator')
    self.parser.add_argument('--mg_n_layers', type=int, default=2,
                             help='number of layers in the multigrid method')
    self.parser.add_argument('--mg_pre_smoothing', type=int, default=2,
                             help='number of pre-smoothing iterations in multigrid')
    self.parser.add_argument('--mg_post_smoothing', type=int, default=2,
                             help='number of post-smoothing iterations in multigrid')

  def parse(self):
    opt = self.parser.parse_args()

    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    image_size_str = '{}x{}'.format(opt.image_size, opt.image_size)
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name, image_size_str)
    if opt.is_train:
      if opt.iterator == 'jacobi' or opt.iterator == 'multigrid':
        # No training needed
        opt.n_epochs = 1

      # Checkpoint name
      if opt.iterator == 'conv':
        iterator_name = '{}{}'.format(opt.iterator, opt.conv_n_layers)
      elif opt.iterator == 'multigrid':
        iterator_name = '{}{}'.format(opt.iterator, opt.mg_n_layers)
      else:
        iterator_name = opt.iterator
      opt.ckpt_name = '{}{}_{}_iter{}_{}_gt{}_{}{:.0e}'.format(\
                          (opt.ckpt_name + '_') if opt.ckpt_name != '' else '',
                          iterator_name, opt.initialization, opt.max_iter_steps,
                          opt.max_iter_steps_from_gt, opt.lambda_gt,
                          opt.optimizer, opt.lr_init)
      opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, image_size_str, opt.ckpt_name)
    else:
      # Note: for testing, dset's and ckpt's image size might be different.
      opt.ckpt_path = opt.load_ckpt_path

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
