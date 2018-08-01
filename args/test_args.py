from .base_args import BaseArgs


class TestArgs(BaseArgs):
  def __init__(self):
    super(TestArgs, self).__init__()

    self.is_train = False
    self.split = 'val'

    # hyperparameters
    self.parser.add_argument('--which_epochs', type=int, nargs='+', default=[-1],
                             help='which epochs to evaluate, -1 to load latest checkpoint')
