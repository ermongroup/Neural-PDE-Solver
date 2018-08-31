import numpy as np
import cv2

def get_geometry(geometry, image_size, batch_size, max_temp):
  if geometry == 'cylinders':
    geom = cylinders
  elif geometry == 'Lshape':
    geom = Lshape
  else:
    raise NotImplementedError

  x, bc, bc_mask = [], [], []
  for i in range(batch_size):
    x_i, bc_i, bc_mask_i = geom(image_size)
    x.append(x_i)
    bc.append(bc_i)
    bc_mask.append(bc_mask_i)
  x = np.stack(x, axis=0) * max_temp
  bc = np.stack(bc, axis=0) * max_temp
  bc_mask = np.stack(bc_mask, axis=0)
  return x, bc, bc_mask

def cylinders(image_size):
  '''
  Return geometry of inner and outer cylinders.
  '''
  c = (image_size + 1) // 2
  r1 = (image_size - 7) // 2
  r2 = int(r1 / 4)

  # Outer cylinder
  x = np.zeros((image_size, image_size)).astype(np.uint8)
  x = cv2.circle(x, (c, c), r1, 255, -1) / 255
  outer = 1 - x
  # Outer shell
#  y = cv2.circle(np.zeros_like(x), (c, c), r1 + 3, 255, -1) / 255
#  outer_shell = y - x

  # Inner cylinder
  center = np.random.randint(-r2 * 1, r2 * 1 + 1, size=2) + c
  center = tuple(center.astype(int))
  inner = cv2.circle(np.zeros_like(x), center, r2, 255, -1) / 255

  bc_mask = inner + outer
  assert np.all(np.logical_or(bc_mask == 0, bc_mask == 1))

  # Boundary values
  half = (image_size + 1) // 2
  v1 = np.random.uniform(0.1, 0.3)
  v2 = np.random.uniform(0.4, 0.6)
  outer[:half, :] = outer[:half, :] * v1
  outer[half:, :] = outer[half:, :] * v2
  bc = outer + inner * 1.0

  # Initialize
  x = np.zeros_like(x)
  x = bc + (1 - bc_mask) * (v1 + v2 + 1) / 3
  return x, bc, bc_mask

def Lshape(image_size):
  '''
  Return L-shape geometry.
  '''
  x = np.random.randint(image_size // 8 * 3, image_size // 8 * 5)
  y = np.random.randint(image_size // 8 * 3, image_size // 8 * 5)
  temperatures = np.concatenate([np.random.uniform(0.5, 1, size=2),
                                 np.random.uniform(0, 0.5, size=2)])

  bc_mask = np.zeros((image_size, image_size))
  bc_mask[:x, :y] = 1
  bc_mask[0, :] = 1
  bc_mask[-1, :] = 1
  bc_mask[:, 0] = 1
  bc_mask[:, -1] = 1

  bc_values = np.zeros((image_size, image_size))
  bc_values[0, :] = temperatures[0]
  bc_values[:, 0] = temperatures[1]
  bc_values[-1, :] = temperatures[2]
  bc_values[:, -1] = temperatures[3]
  # Upper corner
  for i in range(x):
    for j in range(y):
      if i != 0 and j / i < y / x:
        bc_values[i, j] = temperatures[1]
      else:
        bc_values[i, j] = temperatures[0]

  x = np.ones((image_size, image_size)) * temperatures.mean()
  x = x * (1 - bc_mask) + bc_values

  # Random rotate
  k = np.random.randint(4)
  if k > 0:
    x = np.rot90(x, k).copy()
    bc_values = np.rot90(bc_values, k).copy()
    bc_mask = np.rot90(bc_mask, k).copy()

  return x, bc_values, bc_mask
