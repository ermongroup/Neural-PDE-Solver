import torch
import numpy as np

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
