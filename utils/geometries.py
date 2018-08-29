import numpy as np
import cv2

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
  y = cv2.circle(np.zeros_like(x), (c, c), r1 + 3, 255, -1) / 255
  outer_shell = y - x

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
  outer_shell[:half, :] = outer_shell[:half, :] * v1
  outer_shell[half:, :] = outer_shell[half:, :] * v2
  bc = outer_shell + inner * 1.0

  # Initialize
  x = np.zeros_like(x)
  x = bc + (1 - bc_mask) * (v1 + v2 + 1) / 3
  return x, bc, bc_mask
