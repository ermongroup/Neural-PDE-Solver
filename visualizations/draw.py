import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

setting = 'poisson'
width = 6

f = np.load('513x513/{}.npy'.format(setting))[0]
x = f[0] / 100
gt = f[1] / 100
gt = np.pad(gt, width, 'edge')

if setting == 'poisson':
  gt *= 1.4

cmap = plt.cm.viridis
gt = cmap(gt)
size = gt.shape[0]

if setting != 'square':
  bc_values = f[2] / 100
  bc_values = np.pad(bc_values, width, 'edge')
  cmap = plt.cm.viridis
  bc = cmap(bc_values)

if setting == 'square' or setting == 'poisson':
  pass
elif setting == 'centered_Lshape':
  half = 512 // 2 + width
  bc[:(half - width), :(half - width)] = 1
  gt[:(half - width), :(half - width)] = 1
else:
  import cv2
  x = np.zeros((size, size)).astype(np.uint8)
  # Outer cylinder
  r = size // 2 - 1
  c = (size // 2, size // 2)
  x = cv2.circle(x, c, r, 255, -1) / 255
  x = 1 - x
  # Inner cylinder
  c = (309, 202)
  r = 54
  inner = cv2.circle(x.copy(), c, r, 255, -1) / 255
  x += inner
  bc_mask = x[:, :, np.newaxis]
  gt = gt * (1 - bc_mask) + bc_mask

plt.imshow(gt)
plt.show()
#plt.imsave(setting + '_bc.png', bc)
plt.imsave(setting + '_gt.png', gt)
