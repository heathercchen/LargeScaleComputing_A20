from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def plot_time():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  if rank == 0:
    time = np.loadtxt("time_output.txt")

  cores = np.arange(20)+1
  plt.plot(cores, time)
  plt.title("Computation Time against No. of Cores")
  plt.xlabel("No. of cores")
  plt.ylabel("Computation time")
  plt.savefig("ps1_q1.png")

def main():
  plot_time()

if __name__ == '__main__':
  main()