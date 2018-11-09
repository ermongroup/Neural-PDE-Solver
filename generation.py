import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

import utils
from models.iterators import MultigridIterator
from models.heat_model import HeatModel

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str,
                    default=os.path.join(os.environ['HOME'], 'slowbro/PDE/heat'))
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--n_frames', type=int, default=1)
parser.add_argument('--n_runs', type=int, default=1000)
# data
parser.add_argument('--image_size', type=int, default=17)
parser.add_argument('--max_temp', type=int, default=100)
parser.add_argument('--poisson', type=int, default=0)
parser.add_argument('--geometry', type=str, default='square',
                    choices=['square', 'cylinders', 'Lshape',
                             'centered_cylinders', 'centered_Lshape'])
parser.add_argument('--use_model', type=int, default=1,
                    help='Use Multigrid or pretrained model to initialize.')

np.random.seed(666)


def setup_model(opt):
  '''
  Setup the model to initialize the Jacobi solver.
  '''
  use_trained_model = True # Set accordingly
  if opt.image_size < 64:
    depth = 2
    model = MultigridIterator(depth, 8, 8)
  elif opt.image_size < 1024:
    if use_trained_model:
      # Load pretrained UNet model
      model_path = os.path.join(os.environ['HOME'],
                                'slowbro/ckpt/heat/65x65/unet344_random_iter20_0_gt0_adam1e-03')
      model_opt = np.load(os.path.join(model_path, 'opt.npy')).item()
      model_opt.is_train = False
      model_opt.geometry = opt.geometry
      model = HeatModel(model_opt)
      model.load(model_path, 19)
      print('Model loaded from {}'.format(model_path))
    else:
      depth = 4
      model = MultigridIterator(depth, 8, 8)
  else:
    if use_trained_model:
      # Load pretrained UNet model
      model_path = os.path.join(os.environ['HOME'],
                                'slowbro/ckpt/heat/257x257/unet544_random_iter20_0_gt0_adam1e-03')
      model_opt = np.load(os.path.join(model_path, 'opt.npy')).item()
      model_opt.is_train = False
      model_opt.geometry = opt.geometry
      model = HeatModel(model_opt)
      model.load(model_path, 19)
      print('Model loaded from {}'.format(model_path))
    else:
      depth = 6
      model = MultigridIterator(depth, 8, 8)
  if opt.geometry != 'square':
    model.is_bc_mask = True
  return model

def get_solution(x, bc, f):
  '''
  Iterate until error is below a threshold.
  '''
  frames = [x]

  error_threshold = 0.00001
  max_iters = 8000

  error = utils.fd_error(x, bc, f)
  largest_error = error.max().item()
  print('largest error {}'.format(largest_error))
  if largest_error >= error_threshold:
    # Iterate with Jacobi until ground truth
    for i in range(max_iters):
      x = utils.fd_step(x, bc, f)
      error = utils.fd_error(x, bc, f)
      if (i + 1) % 100 == 0:
        largest_error = error.max().item() # largest error in the batch
        print('Iter {}: largest error {}'.format(i + 1, largest_error))
        if largest_error < error_threshold:
          break

  # Add ground truth to frames
  y = x.cpu().numpy()
  frames.append(y)
  frames = np.stack(frames, axis=1) # batch_size x (n_frames + 1) x image_size x image_size
  return frames

def get_heat_source(image_size, batch_size):
  '''
  Return a Gaussian heat source.
  '''
  heat_src = []
  for i in range(batch_size):
    scale = np.random.uniform(3, 3.5) / (image_size ** 2)
    f = - utils.gaussian(image_size - 2) * scale # Negative
    f = torch.Tensor(f).unsqueeze(0)
    f = utils.pad_boundary(f, torch.zeros(1, 4)) # (1 x image_size x image_size)
    heat_src.append(f)
  heat_src = torch.cat(heat_src, dim=0)
  if torch.cuda.is_available():
    heat_src = heat_src.cuda()
  return heat_src

def generate_square(opt):
  '''
  Generate data with square boundary conditions.
  Multiply max_temp in the end!
  '''
  if opt.poisson:
    dir_name = 'poisson_{}x{}'.format(opt.image_size, opt.image_size)
  else:
    dir_name = '{}x{}'.format(opt.image_size, opt.image_size)
  opt.save_dir = os.path.join(opt.save_dir, opt.geometry, dir_name)
  frame_dir = os.path.join(opt.save_dir, 'frames')
  os.makedirs(frame_dir, exist_ok=True)

  # shape: n_runs x batch_size x 4
  boundary_conditions = np.random.rand(opt.n_runs, opt.batch_size, 4)
  if opt.poisson:
    # Lower temperature
    boundary_conditions *= 0.75
  # Save opt
  np.save(os.path.join(opt.save_dir, 'opt.npy'), opt)

  if opt.use_model:
    # Setup Multigrid model
    model = setup_model(opt)

  for run in range(opt.n_runs):
    x = np.random.rand(opt.batch_size, opt.image_size, opt.image_size)
    bc = boundary_conditions[run]
    x = utils.set_boundary(x, bc)
    original_x = x.copy()[:, np.newaxis, :, :]
    if opt.poisson:
      f = get_heat_source(opt.image_size, opt.batch_size)
    else:
      f = None

    x = torch.Tensor(x)
    bc = torch.Tensor(bc)
    if torch.cuda.is_available():
      x = x.cuda()
      bc = bc.cuda()

    # Initialize with average of boundary conditions
    x[:, 1:-1, 1:-1] = bc.mean(dim=1).view(-1, 1, 1)

    if opt.use_model:
      # Use Multigrid model to initialize
      for i in range(400):
        x = model.iter_step(x, bc, f).detach()

    # Find solution
    frames = get_solution(x, bc, f)
    frames = np.concatenate([original_x, frames[:, 1:2]], axis=1)
    assert frames.shape[1] == opt.n_frames + 1

    if f is not None:
      # Concatenate f to frames.
      f = f.cpu().numpy()[:, np.newaxis, :, :]
      frames = np.concatenate([frames, f], axis=1)
      # Check values < 1
      max_value = frames.reshape((opt.batch_size, 3, opt.image_size ** 2))[:, 1, :]\
                    .max(axis=1)
      scaling = np.ones(opt.batch_size)
      scaling[max_value >= 1] = 0.99 / max_value[max_value >= 1]
      # Scale the ones that has values > 1
      frames *= scaling[:, np.newaxis, np.newaxis, np.newaxis]
      boundary_conditions[run] *= scaling[:, np.newaxis]

    assert np.all(frames[:, 1, :, :] <= 1 + 1e-5)

    frames = frames * opt.max_temp

    np.save(os.path.join(frame_dir, '{:04d}.npy'.format(run)), frames)
    print('Run {} saved.'.format(run))

  # Save boundaries
  boundary_conditions *= opt.max_temp
  np.save(os.path.join(opt.save_dir, 'bc.npy'), boundary_conditions)

def generate_geometry(opt):
  '''
  Generate data with geometry.
  Multiply max_temp in the end!
  '''
  assert opt.poisson == 0 # f = 0 for now
  opt.save_dir = os.path.join(opt.save_dir, opt.geometry,
                              '{}x{}'.format(opt.image_size, opt.image_size))
  frame_dir = os.path.join(opt.save_dir, 'frames')
  os.makedirs(frame_dir, exist_ok=True)
  # Save opt
  np.save(os.path.join(opt.save_dir, 'opt.npy'), opt)

  if opt.use_model:
    # Setup Multigrid model
    model = setup_model(opt)

  for run in range(opt.n_runs):
    x, bc_values, bc_mask = utils.get_geometry(opt.geometry, opt.image_size,
                                               opt.batch_size, 1)
    bc = np.stack([bc_values, bc_mask], axis=1) # batch_size x 2 x image_size x image_size
    x = torch.Tensor(x)
    original_x = x.numpy().copy()[:, np.newaxis, :, :]
    bc = torch.Tensor(bc)
    f = None
    if torch.cuda.is_available():
      x = x.cuda()
      bc = bc.cuda()

    if opt.use_model:
      # Use Multigrid model to initialize
      for i in range(400):
        x = model.iter_step(x, bc, f).detach()

    # Find solution
    frames = get_solution(x, bc, f)
    assert frames.shape[1] == 2
    # Add bc and mask
    data = np.concatenate([original_x, frames[:, 1:2],
                           bc_values[:, np.newaxis, :, :],
                           bc_mask[:, np.newaxis, :, :]], axis=1)
    data[:, :3] *= opt.max_temp
    np.save(os.path.join(frame_dir, '{:04d}.npy'.format(run)), data)
    print('Run {} saved.'.format(run))

if __name__ == '__main__':
  opt = parser.parse_args()
  assert (opt.image_size - 1) % 16 == 0, 'image_size must be 2^n + 1'
  if opt.geometry == 'square':
    generate_square(opt)
  else:
    generate_geometry(opt)
