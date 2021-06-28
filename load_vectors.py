import numpy as np


def load_npy(file_name):
  """
  Loads word vector set from .npy file and .txt list of words
  Both files must have same name (different extension).
  Input:
    - file_name: file path (without extension)
  """
    
  fname_txt = file_name + '.txt'
  fname_npy = file_name + '.npy'

  # load npy and txt
  npy = np.load(fname_npy)
  txt = open(fname_txt, encoding='utf-8').read().splitlines()  

  # index str
  w2ind = dict()
  for ind, wd in enumerate(txt):
      w2ind[wd] = ind

  # w2v function
  def w2v(wd):
    default_vector = np.zeros(len(npy[0]))
    try:
      return npy[w2ind[wd]]
    except KeyError:
#      print (wd + ' not found in vecset')
      return default_vector
  
  wd2vect = w2v
  wd2vect.name = file_name
  wd2vect.dim = len(npy[0])

  return wd2vect
