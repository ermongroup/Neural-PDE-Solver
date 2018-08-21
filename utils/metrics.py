import numpy as np

from .misc import to_numpy

class Metrics(object):
  def __init__(self, scale, error_threshold):
    print('Runtime scaling: {}'.format(scale))
    self.scale = scale
    self.error_threshold = error_threshold
    self.model_steps = []
    self.fd_steps = []
    self.invalid = 0

  def update(self, error_dict):
    errors = to_numpy(error_dict['model errors'])
    fd_errors = to_numpy(error_dict['Jacobi errors'])
    batch_size, length = errors.shape
    for i in range(batch_size):
      if np.all(fd_errors[i] >= self.error_threshold):
        self.invalid += 1
        continue
      model_step = np.nonzero(errors[i] < self.error_threshold)[0][0]
      fd_step = np.nonzero(fd_errors[i] < self.error_threshold)[0][0]
      model_step *= self.scale # scaling for model
      self.model_steps.append(model_step)
      self.fd_steps.append(fd_step)

  def get_results(self):
    print('Invalid: {}/{}'.format(self.invalid, len(self.fd_steps) + self.invalid))
    self.model_steps = np.array(self.model_steps)
    self.fd_steps = np.array(self.fd_steps)
    ratios = self.model_steps / self.fd_steps
    results = {'Jacobi': self.fd_steps.mean(),
               'model': self.model_steps.mean(),
               'ratio': ratios.mean()}
    return results

  def reset(self):
    self.model_steps = []
    self.fd_steps = []
    self.invalid = 0
