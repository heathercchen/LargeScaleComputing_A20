import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cltools
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt
import time
import rasterio

def sim_gpu_nvdi():
  band4 = rasterio.open('LC08_B4.tif') #red
  band5 = rasterio.open('LC08_B5.tif') #nir

  #Convert nit and red objects to float64 arrays
  red = band4.read(1).astype('float64')
  nir = band5.read(1).astype('float64')

  #Get the computation time using original serial solution
  t0_s = time.time()
  nvdi_s = (nir - red) / (nir + red)
  final_time_s = time.time()
  serial_time = final_time_s - t0_s

  #Plot the graph using serial solution
  plt.imsave('ps_q3_serial_ori.png', nvdi_s)

  #Get the computation time using gpu
  #Set up OpenCL context and command queue
  t0_g = time.time()
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx) 

  red_dev = cl_array.to_device(queue, red)
  nir_dev = cl_array.to_device(queue, nir)

  nvdi_comb = ElementwiseKernel(ctx,"double *x, double *y, double *res","res[i] = (x[i] - y[i]) / (x[i] + y[i])")

  res_gpu = cl.array.empty_like(nir_dev)
  nvdi_comb(nir_dev, red_dev, res_gpu)
  nvdi_gpu = res_gpu.get()
  
  final_time_g = time.time()
  gpu_time = final_time_g - t0_g

  #Plot the graph using gpu solution (to prove that they are the same)
  plt.imsave('ps_q3_gpu_ori.png', nvdi_gpu)

  #Report the time for serial solution and gpu
  print('The time using serial solution: {0:.4f} seconds'.format(serial_time))
  print('The time using gpu: {0:.4f} seconds'.format(gpu_time))

def main():
  sim_gpu_nvdi()

if __name__ == '__main__':
  main()