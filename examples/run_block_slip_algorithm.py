import copy
import numpy as np
import time
import matplotlib.pyplot as plt
import joblib
import psutil
import os

from utility import *
import trs4slip
from run_slip_algorithm import slip

class ActiveSet:
    def __init__(self):
        self.data = dict()
    def append_data(self, k, patchNumber, w, ared):
        if (k, patchNumber) not in self.data: #set of k, patch
            self.data[(k, patchNumber)] = []
        self.data[(k, patchNumber)].append((w, ared)) #store solution, ared
    def compute_max(self):
        k = 0
        if len(self.data)==0:
          return -np.inf,[]
        else:
          maxval = np.zeros(len(self.data))
          for key, value in self.data.items():
            maxval[k] = value[0][1]
            k += 1
          ind = np.argmax(maxval)
          return maxval[ind],list(self.data.keys())[ind]
    def remove_data(self, key):
        self.data.pop(key)

class WorkingSet:
    def __init__(self, npatches):
        self.data = {(k,D) for k,D in zip([0]*npatches, range(1, npatches+1))}
    def append_data(self, k, patchNumber):
        self.data.add((k, patchNumber))
    def remove_data(self, k, patchNumber):
        self.data.remove((k, patchNumber))
    def getActivePatches(self):
        aP = []
        for e in self.data:
            aP.append(e[1])
        return aP

class OneDPatches:
  def __init__(self, di, numblocks=2, customblocks=False, buffer=4, *args): #put in grid axes
      self.idx = dict()
      # self.extremaPts = dict()
      if ~customblocks:
        blkSize_noB = int(np.floor(di.N/numblocks))
        for j in range(0, numblocks):
          if j == 0:
             self.idx[(j+1)] = np.arange(0, (blkSize_noB+buffer))
          elif j == (numblocks-1):
             self.idx[(j+1)] = np.arange(di.N-(blkSize_noB+buffer), di.N)
          else:
             self.idx[(j+1)] = np.arange(j*blkSize_noB-buffer, (j+1)*blkSize_noB+buffer)
          # self.extremaPts[(i, j)]

def patchUpdate(ind, gn, xn, lb, ub, lo_bangs, Drad, alpha, N, i):
    #temp patch variable
    w = np.zeros(len(ind), dtype=np.int32)
    M = lo_bangs.shape[0]
    ## buffers for c++ vector initialization
    D0 = N // 4
    vert_costs_buffer  = np.empty(N*M*(D0 + 1) + 2)
    vert_layer_buffer  = np.empty(N*M*(D0 + 1) + 2, dtype=np.int32)
    vert_value_buffer  = np.empty(N*M*(D0 + 1) + 2, dtype=np.int32)
    vert_prev_buffer   = np.empty(N*M*(D0 + 1) + 2, dtype=np.int32)
    vert_remcap_buffer = np.empty(N*M*(D0 + 1) + 2, dtype=np.int32)

    trs4slip.run(#can simply put patch idx, check xnk
        w,
        gn / alpha,
        xn,
        lo_bangs,
        Drad,
        vert_costs_buffer,
        vert_layer_buffer,
        vert_value_buffer,
        vert_prev_buffer,
        vert_remcap_buffer,
        True,
        lb,
        ub,
    )
    print('finish')
    return w, i


def eval_tv(x):
    return np.sum(np.abs(x[1:] - x[:-1]))

def blockslip(x0, patches, lo_bangs, alpha, h, Delta0, sigma, maxiter, maxiter_k, tol, lg_cm, di, f_vec, useParallel = False):
    assert x0.ndim == 1
    assert lo_bangs.ndim == 1
    # N, M = x0.shape[0], lo_bangs.shape[0]

    xn = copy.deepcopy(x0) #make smaller so subproblem solver generates correct size
    ## initialize temp variables
    # xnk = copy.deepcopy(x0); #np.empty((N,), dtype=np.int32)
    LnablaF = 1e1 ## guess?
    npatches = len(patches.idx)


    print("n - Iter  k - Iter  Patch     fnki          tvnki      J(x)")
    #Outer total loop
    for n in range(maxiter):
        A = ActiveSet()
        W = WorkingSet(npatches)
        fn = lg_objective_var(xn, lg_cm, di, f_vec) #eval_f(xn)
        gn = lg_jacobian_var(xn, lg_cm, di, f_vec) #eval_jac(xn)
        tvn = eval_tv(xn)
        #Inner patch loop
        for k in range(maxiter_k):
          # while not bool(W) or k == 0: #empty list returns false
          activePatches = W.getActivePatches()
          Drad          = Delta0*(2**-k)

          #subproblem solve only until we can figure out data sharing
          if useParallel:
            results = joblib.Parallel(n_jobs = npatches, backend='multiprocessing')(
            joblib.delayed(patchUpdate)(patches.idx[i], gn[patches.idx[i]], xn[patches.idx[i]], xn[patches.idx[i][0]], xn[patches.idx[i][-1]],lo_bangs, Drad, alpha, xn.shape[0], i) for i in activePatches)
          else:
            results = []
            for i in activePatches:
              ind = patches.idx[i]
              results.append(patchUpdate(ind, gn[ind], xn[ind], xn[ind[0]], xn[ind[-1]], lo_bangs, Drad, alpha, xn.shape[0], i))

          for (w,i) in results:
            ind = patches.idx[i]
            xnk_temp = xn
            xnk_temp[ind] = w
            fnk = lg_objective_var(xnk_temp, lg_cm, di, f_vec)
            tvnk = eval_tv(xnk_temp)

            ared_nkd = fn - fnk + alpha * tvn - alpha * tvnk
            pred_nkd = gn.dot(xn - xnk_temp) + alpha * tvn - alpha * tvnk
            print("%4u     %4u     %4u     %.3e      %.3e   %.3e" % (n, k, i, fnk, alpha*tvnk, fn + alpha*tvn))
            pred_positive = sigma*pred_nkd > -1.0e-6
            if ared_nkd >= sigma * pred_nkd and pred_positive:
              A.append_data(k, i, w, ared_nkd)
            elif pred_positive and A.compute_max()[0] < pred_nkd + LnablaF*Drad:
              W.append_data(k+1, i)
            #remove k, D
            W.remove_data(k, i)
          #   patchUpdate(patches.idx[i], gn, xn, lo_bangs, Drad, alpha, A, W, LnablaF, sigma, lg_cm, di, f_vec, fn, tvn, n, k, i)
          #looper = asyncio.gather(*[patchUpdate(patches.idx(i), gn, xn, lo_bangs, Drad, A, W) for i in activePatches]
          if not bool(W):
             break

        if len(A.data)==0:
            print("Set of Active patches is empty!")
            return xn

        txn = xn.copy()
        j0  = fn + alpha*tvn

        while len(A.data)!=0:
            maxKey     = A.compute_max()[1]
            temp_x     = A.data[maxKey][0][0]
            ind        = patches.idx[maxKey[1]] #shoulld just pick out grid indices
            txn[ind]   = temp_x
            jnt        =  lg_objective_var(txn, lg_cm, di, f_vec) + alpha*eval_tv(txn)
            if jnt < j0:
              xn[ind] = txn[ind] #should be all you need.
              j0 = jnt
              A.remove_data(maxKey)
            else:
              break

        print("%4u       ----      ----      ----         ----      %.3e" % (n, j0))
        if n == maxiter - 1:
          print("Iteration limit (%d) reached. Solution may be instationary." % (maxiter))
          break
        stopCrit = np.linalg.norm(gn[np.insert((xn[1:] - xn[:-1]) !=0, 0, False)])/h
        if np.abs(stopCrit) < tol:
          print("Stationarity Condition reached. \sum_{i\in n_s} ||g_i||_2  =  ", stopCrit)
          break
    return xn



def lg_objective_var(x, lg_cm, di, f_vec):
    return lg_objective(x, lg_cm, di, f_vec)

def lg_jacobian_var(x, lg_cm, di, f_vec):
    return lg_jacobian(x, lg_cm, di, f_vec)

def main(N=2**12, alpha = 5e-5, numPatches=5, tol = 1e-4, usePlots = True):
    # N = 2**14 #32768 #16384 #8192
    di = create_discretization_info(-1., 1., N, 5)

    lo_bangs = np.array([-1, 0, 1], dtype=np.int32)

    state_vec = np.zeros((N,))
    control_vec = np.zeros((N,))

    # == Setup convolution ==
    # * at Legendre-Gauss points for exact evaluation
    lg_int_mda_mat = lg_int_mda(di)
    lg_cm = LgConvPwcMat(lg_int_mda_mat, di)
    # == Setup optimization ==
    # * regularization
    # alpha = 5e-5
    # * initial value
    x = np.zeros(N, dtype=np.int32)
    # * desired state for tracking objective
    f_vec = f(di.lg_t_vec)
    # * function handles
    eval_f = lambda x: lg_objective_var(x, lg_cm, di, f_vec)
    eval_jac = lambda x: lg_jacobian_var(x, lg_cm, di, f_vec)
    # * algorithm control
    Delta0 = N // 4
    sigma = 1e-3
    maxiter = 100
    maxiter_k = 20
    h = 2./N

    # numPatches = 4
    bufferSize = int(np.floor(N/(numPatches*2)))
    patches = OneDPatches(di, numPatches, buffer=bufferSize)
    # == Optimization with convolution evaluated at Legendre-Gauss points ==
    opt_start = time.time()
    x_bs = blockslip(x, patches, lo_bangs, alpha, h, Delta0, sigma, maxiter, maxiter_k, tol, lg_cm, di, f_vec)
    timebs = time.time() - opt_start
    fbs = eval_f(x_bs)
    tvbs = eval_tv(x_bs)

    ## would probably have to put below into slip
    # process = psutil.Process(os.getpid())
    # memoryUsage = process.memory_info().rss
    if np.log(N)/np.log(2) <= 15: #(memoryUsage*1e-9)>2: #check if memory usage is too high
      opt_start = time.time()
      x_s  = slip(eval_f, eval_jac, x, lo_bangs, alpha, h, Delta0, sigma, maxiter)
      time_s = time.time() - opt_start
      state_vec_bs = lg_cm.conv(x_bs)
      state_vec_s  = lg_cm.conv(x_s)
      control_vec  = x_bs
      print("N = ", N, "alpha = ", alpha, "Num Patch = ", numPatches, "||x_bs - x_s|| = ",       np.linalg.norm(x_bs - x_s), "||S(x_bs) - S(x_s)|| = ", np.linalg.norm(state_vec_bs - state_vec_s))
      fs = eval_f(x_s)
      tvs = eval_tv(x_s)
    else:
      print('Could not run slip')
      state_vec_bs = lg_cm.conv(x_bs)
      x_s = np.empty(x_bs.shape)
      x_s[:] = np.nan
      state_vec_s  = lg_cm.conv(x_s)
      fs = eval_f(x_s)
      tvs = eval_tv(x_s)
      time_s = np.nan
    if usePlots:
      plt.subplot(1, 2, 1)
      ind = np.linspace(0., 1., N)
      plt.plot(ind, control_vec)
      plt.xlabel('Control')
      ax = plt.gca()
      ax.set_xlim([0., 1.])
      ax.set_ylim([-1.15, 1.15])
      #plt.ylabel('Integer - value')
      for i in patches.idx:
        pb = 0.02
        terms = patches.idx[i][[0,-1]]
        plt.hlines(pb*np.mod(i,2) - pb - 1.05, ind[terms[0]], ind[terms[1]], colors = 'k', lw=.5, linestyle = 'dotted')
        if np.mod(i,2)==0:
          plt.vlines(ind[terms[0]], -1.05, -2*pb-1.05, colors = 'k', lw = .5)
          plt.vlines(ind[terms[1]], -1.05, -2*pb-1.05, colors = 'k', lw = .5)
          plt.text(ind[int(np.median(patches.idx[i]))], -4*pb-1.05, str(i), fontsize=6)
        else:
          plt.vlines(ind[terms[0]], pb-1.05, -1.05-pb, colors = 'k', lw = .5)
          plt.vlines(ind[terms[1]], pb-1.05, -1.05-pb, colors = 'k', lw = .5)
          plt.text(ind[int(np.median(patches.idx[i]))], .5*pb-1.05, str(i), fontsize=6)

      plt.subplot(1, 2, 2)
      plt.plot(np.linspace(0., 1., state_vec_s.shape[0]),   state_vec_bs)
      plt.xlabel('State')
      ax = plt.gca()
      ax.set_xlim([0., 1.])
      ax.set_ylim([-.015, 0.015])
      plt.tight_layout()
      plt.savefig(str(N)+"_"+str(alpha)+"_"+str(numPatches), format = 'eps', dpi=1200)
      plt.close()
    return (fbs, tvbs, fs, tvs, timebs, time_s), (control_vec, state_vec_bs)

if __name__ == "__main__":
    main()



