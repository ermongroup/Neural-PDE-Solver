from .iterators import *

def get_iterator(opt):

  if opt.iterator == 'jacobi':
    iterator = JacobiIterator().cuda()
  elif opt.iterator == 'multigrid':
    iterator = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                 opt.mg_post_smoothing).cuda()
  elif opt.iterator == 'cg':
    iterator = ConjugateGradient(opt.cg_n_iters)
  elif opt.iterator == 'conv':
    iterator = ConvIterator(opt.activation, opt.conv_n_layers).cuda()
  elif opt.iterator == 'unet':
    iterator = UNetIterator(opt.activation, opt.mg_n_layers,
                            opt.mg_pre_smoothing, opt.mg_post_smoothing).cuda()
  else:
    raise NotImplementedError

  # Compare to Jacobi methods
  if opt.iterator == 'conv' or opt.iterator == 'multigrid':
    compare_model = JacobiIterator()
  elif opt.iterator == 'unet' or opt.iterator == 'cg':
    compare_model = MultigridIterator(opt.mg_n_layers, opt.mg_pre_smoothing,
                                      opt.mg_post_smoothing)
  else:
    compare_model = None
  # ratio of operations
  if compare_model is not None:
    operations_ratio = iterator.n_operations / compare_model.n_operations
  else:
    operations_ratio = 1

  return iterator, compare_model, operations_ratio
