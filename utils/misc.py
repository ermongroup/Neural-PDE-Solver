import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def prompt_yes_no(question):
  i = input(question + ' [y/n]: ')
  if len(i) > 0 and (i[0] == 'y' or i[0] == 'Y'):
    return True
  else:
    return False

def blue(string):
  return '\033[94m'+string+'\033[0m'

def yellow(string):
  return '\033[93m'+string+'\033[0m'

def red(string):
  return '\033[91m'+string+'\033[0m'

def plot(data_list):
  '''
  data: A list of dictionaries.
  return: numpy array of image
  '''
  for data in data_list:
    y = data['y']
    x = np.arange(len(y))
    plt.plot(x, y, label=data['label'])
  plt.xlabel('iterations')
  plt.legend()
  plt.savefig('tmp.png')
  plt.close()
  # Read in tmp.png
  img = np.array(Image.open('tmp.png')) # H x W x 4
#  img = img[:, :, :3].transpose((2, 0, 1)) / 255 # 3 x H x W
  return img
