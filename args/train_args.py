from .base_args import BaseArgs


class TrainArgs(BaseArgs):
  def __init__(self):
    super(TrainArgs, self).__init__()

    self.is_train = True
    self.split = 'train'

    self.parser.add_argument('--n_epochs', type=int, default=200, help='total # of epochs')
    self.parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    self.parser.add_argument('--lr_init', type=float, default=2e-4, help='initial learning rate')
    self.parser.add_argument('--lr_decay_start', type=int, default=100, help='epoch at which lr decay starts')
    self.parser.add_argument('--load_ckpt_dir', type=str, default='', help='directory of checkpoint')

    # Hyperparameters
    self.parser.add_argument('--lambda_gt', type=float, default=0)
