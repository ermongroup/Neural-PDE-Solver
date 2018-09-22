from .iterators import *

def get_iterator(opt):
  is_train = opt.is_train

  if opt.iterator == 'jacobi':
    iterator = JacobiIterator()
    is_train = False
  elif opt.iterator == 'multigrid':
    iterator = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                 opt.mg_post_smoothing)
    is_train = False
  elif opt.iterator == 'cg':
    iterator = ConjugateGradient(opt.cg_n_iters)
    is_train = False
  elif opt.iterator == 'conv':
    iterator = ConvIterator(opt.activation, opt.conv_n_layers)
  elif opt.iterator == 'unet':
    iterator = UNetIterator(opt.activation, opt.mg_n_layers,
                            opt.mg_pre_smoothing, opt.mg_post_smoothing)
  else:
    raise NotImplementedError

  if torch.cuda.is_available():
    iterator = iterator.cuda()

  # Compare to Jacobi methods
  if opt.iterator == 'conv' or opt.iterator == 'multigrid':
    compare_model = JacobiIterator()
  elif opt.iterator == 'unet' or opt.iterator == 'cg':
    compare_model = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                      opt.mg_post_smoothing)
  else:
    compare_model = None

  if opt.geometry != 'square':
    # Set is_bc_mask to True if geometry is not square
    iterator.is_bc_mask = True
    if compare_model is not None:
      compare_model.is_bc_mask = True

  # ratio of operations
  if compare_model is not None:
    operations_ratio = iterator.n_operations / compare_model.n_operations
  else:
    operations_ratio = 1

  return iterator, compare_model, operations_ratio, is_train
