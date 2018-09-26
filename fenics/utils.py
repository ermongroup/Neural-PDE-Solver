import numpy as np
import os

def rms(x, gt):
  return np.sqrt(((x - gt) ** 2).mean())

def get_boundary_conditions(geometry, num, dset_path, max_temp):
  '''
  Get data.
  '''
  if geometry == 'square':
    n = num // 16 + 1
    bc = np.load(os.path.join(dset_path, 'bc.npy'))[:n] / max_temp
    bc = bc.reshape((-1, 4))[:num]
    return bc

  elif geometry == 'centered_cylinders':
    bcs = []
    for i in range(num):
      b = np.random.uniform(0.2, 0.8)
      bcs.append([b, 1])
    bc = np.array(bcs)
    print(bc.shape)
    return bc

  elif geometry == 'centered_Lshape':
    bcs = []
    for i in range(num):
      b = np.concatenate([np.random.uniform(0.5, 1, size=2),
                          np.random.uniform(0, 0.5, size=2)])
      bcs.append(b)
    bc = np.array(bcs)
    return bc
