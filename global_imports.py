import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import itertools


# generic useful functions which don't fit anywhere else

''' simple class to time things easily '''
class Stopwatch():

	def __init__(self):
		self.t_start = None

	def start(self):
		self.t_start = time.time()

	def stop(self, print_result=True):
		if self.t_start is not None:
			delta_t = time.time() - self.t_start
			self.t_start = time.time()
			if print_result:
				print(f"timer: {delta_t}s")

			return delta_t
		
		return 0


''' get indices of all occurences of element in list
	(taken from https://thispointer.com/python-how-to-find-all-indexes-of-an-item-in-a-list/) '''
def get_indices_of_value(array, value):
    pos_list = []
    idx_start = 0
    while True:
        try:
            idx_start = array.index(value, idx_start)
            pos_list.append(idx_start)
            idx_start += 1
        except ValueError as e:
            break
    return tuple(pos_list)