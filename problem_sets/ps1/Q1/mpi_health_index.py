from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import time

def sim_health_index(n_runs):
  #Get the number of processors and the rank of different processors
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  #Start time
  t0 = time.time()

  #Specify the number of runs on each processors
  N = int(n_runs/size)

  #Set model parameters
  rho = 0.5
  mu = 3.0
  sigma = 1.0
  z_0 = mu

  #Set simulation parameters, draw all idiosyncratic random shocks, and create empty containers
  T = int(4160) #Set the number of periods for each simulation
  np.random.seed(25)
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T,N))
  z_mat = np.zeros((T, N))

  for s_ind in range(N):
    z_tm1 = z_0
    for t_ind in range(T):
      e_t = eps_mat[t_ind, s_ind]
      z_t = rho * z_tm1 + (1 - rho) * mu + e_t
      z_mat[t_ind, s_ind] = z_t
      z_tm1 = z_t
  z_mat_array = np.array(z_mat)

  z_mat_all = None
  if rank == 0:
    z_mat_all = np.empty([N*size, T], dtype='float')
  comm.Gather(sendbuf = z_mat_array, recvbuf = z_mat_all, root=0)

  if rank == 0:
    final_time = time.time()
    time_elapsed = final_time - t0

    with open('time_output.txt', 'a') as f:
      print(time_elapsed, file=f)

  return

def main():
    sim_health_index(n_runs = 1000)

if __name__ == '__main__':
    main()