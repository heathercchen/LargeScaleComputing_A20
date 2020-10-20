from mpi4py import MPI
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import time

def sim_maxi_rho(n_runs):
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  N = int(n_runs)

  #Start time
  t_0 = time.time()

  #Generate the matrix and rhos to distribute on 0 processor
  T = int(4160) #Set the number of periods for each simulation
  np.random.seed(25)
  mu = 3.0
  sigma = 1.0
  z_0 = mu

  #Scatter different set of rhos to different processors
  numPerRank = int(200/size)
  rhos = None
  if rank == 0:
    rhos = np.linspace(-0.95, 0.95, 200)
  
  recv_rhos = np.empty(numPerRank, dtype='float')
  comm.Scatter(rhos, recv_rhos, root=0)
  #Check the results
  #print('Rank: ',rank, ', data received: ', recv_rhos)

  #Broadcast the shock matrix to each processor
  if rank == 0:
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T,N))
  else:
    eps_mat = np.empty([T, N], dtype='float')
  
  comm.Bcast(eps_mat, root=0)

  #Find the optimal rho in each processor
  #Create a dictionary to store the optimal rho, N and t
  mat_avg = {}
  for rho in recv_rhos:
    #Reset z_0 and z_mat for each rho
    z_0 = mu
    z_mat = np.zeros((T, N))
    res_ts = [] #A list to store results of t
    for n_ind in range(N):
      z_tm1 = z_0
      for t_ind in range(T):
        e_t = eps_mat[t_ind, n_ind]
        z_t = rho * z_tm1 + (1-rho) * mu + e_t
        if z_t < 0:
          #Get the optimal 
          res_ts.append(t_ind)
          break
        z_mat[t_ind, n_ind] = z_t
        z_tm1 = z_t
    
    #print("Results of t for", rho, ":", res_ts)
    avg = sum(res_ts)/len(res_ts)
    #Store the average number of t for this specific rho into the dictionary
    mat_avg[rho] = avg
    #print(mat_avg)
  
  #Gather the average ts to processor 0
  result = mat_avg.items()
  avg_list = np.array(list(result))
  avg_list_total = None
  if rank == 0:
    avg_list_total = np.empty([200, 2], dtype='float')

  comm.Gather(avg_list, avg_list_total, root=0)
  if rank == 0:
    #Get the computation time
    final_time = time.time()
    time_elapsed = final_time - t_0
    print("Computation Time: {0:.4f} seconds".format(time_elapsed))

    #Plot each average t to corresponding rho
    res_df = pd.DataFrame(avg_list_total, columns=['rho', 't'])
    plt.plot(res_df['rho'], res_df['t'])
    plt.title("Average periods t to the first negative z_t for different rhos")
    plt.xlabel('rho')
    plt.ylabel('average period t')
    plt.savefig("ps1_q2.png")
    
    #Get the optimal (maximum) t and its corresponding rho
    max_index = res_df['t'].argmax()
    optimal_rho = res_df['rho'][max_index]
    optimal_t = res_df['t'][max_index]
    print("Optimal rho: {0:.4f}".format(optimal_rho), "Corresponding t: {0:.4f}".format(optimal_t))
  
  return

def main():
  sim_maxi_rho(1000)

if __name__ == '__main__':
  main()