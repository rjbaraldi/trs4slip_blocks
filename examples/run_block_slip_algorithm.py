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

def eval_tv(x):
    return np.sum(np.abs(x[1:] - x[:-1])) + np.abs(x[0]) + np.abs(x[-1])


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

class PatchProblem:
  def __init__(self, ind, gn, xn, lb, ub, lo_bangs, alpha, N, i):
      self.ind = ind
      self.gn  = gn
      self.xn  = xn
      self.lb  = lb
      self.ub  = ub
      self.lo_bangs = lo_bangs
      self.alpha    = alpha
      self.N        = N
      self.i        = i
      self.temp     = np.ones((len(ind)- 1,))

  def patchUpdate(self, Drad, i):
      w = np.zeros(len(self.ind), dtype=np.int32)
      M = self.lo_bangs.shape[0]
      number = time.time()
      rand = np.random.randint(0, 1e6)
      fn = "%d_%d.npz" % (number, rand)

      trs4slip.run_top(
        w,
        self.gn / self.alpha,
        self.xn,
        self.lo_bangs,
        self.temp,
        Drad,
        True,
        self.lb,
        self.ub,
        1.,
        1.
      )

      return w, self.i




def blockslip(x0, patches, lo_bangs, alpha, h, Delta0, sigma, maxiter, maxiter_k, tol, lg_cm, di, f_vec, useParallel = False):
    assert x0.ndim == 1
    assert lo_bangs.ndim == 1
    N, M = x0.shape[0], lo_bangs.shape[0]

    xn = copy.deepcopy(x0) #make smaller so subproblem solver generates correct size

    LnablaF = 1e1 ## guess? max of abs of gn? can we compute this for test problem?
    npatches = len(patches.idx)

    Delta0_start = x0.shape[0] / ((npatches + 1) / 2)
    while Delta0 / 2 >= Delta0_start:
      Delta0 /= 2

    print("n - Iter  k - Iter  Patch     fnki          tvnki      J(x)")
    #Outer total loop
    for n in range(maxiter):
        A = ActiveSet()
        W = WorkingSet(npatches)
        fn = lg_objective_var(xn, lg_cm, di, f_vec) #eval_f(xn)
        gn = lg_jacobian_var(xn, lg_cm, di, f_vec) #eval_jac(xn)

        v1 = gn[np.insert((xn[1:] - xn[:-1]) !=0, 0, False)]
        v2 = gn[np.append((xn[1:] - xn[:-1]) !=0, [False])]
        stop_crit = np.linalg.norm(.5 * (v1 + v2)) / h
        print("Instationarity = %.2e" % stop_crit)
        if n > 0 and stop_crit < 1e-6:
          print("Instationarity = %.2e < 1e-6." % stop_crit)
          break

        tvn = eval_tv(xn)
        Drad = Delta0

        #Inner patch loop
        for k in range(maxiter_k):
          # get list of active patches
          activePatches = W.getActivePatches()
          # determine set of patch problems
          pProbs = [PatchProblem(patches.idx[i], gn[patches.idx[i]], xn[patches.idx[i]], xn[patches.idx[i][0]], xn[patches.idx[i][-1]], lo_bangs, alpha, xn.shape[0], i) for i in activePatches]
          # determine delta
          Drad          = Delta0*(2**-k)

          if len(activePatches) == 0 or Drad < 1:
            break

          #subproblem solve only until we can figure out data sharing
          if useParallel:
            results = joblib.Parallel(n_jobs = npatches, backend='multiprocessing')(
            joblib.delayed(pProbs[i].patchUpdate(Drad, i) for i in range(0, len(activePatches))))
          else:
            results = []
            for i in range(0, len(activePatches)):
              ind = pProbs[i].ind
              lidx = ind[0] - 1
              ridx = ind[-1] + 1
              results.append(pProbs[i].patchUpdate(Drad, i)
              )

          xnk_temp = copy.deepcopy(xn)

          for (w,i) in results:
            ind = patches.idx[i]
            xnk_temp[ind] = w
            fnk = lg_objective_var(xnk_temp, lg_cm, di, f_vec)
            tvnk = eval_tv(xnk_temp)

            ared_nkd = fn - fnk + alpha * tvn - alpha * tvnk
            pred_nkd = gn.dot(xn - xnk_temp) + alpha * tvn - alpha * tvnk

            print("%4u     %4u     %4u     %.3e      %.3e   %.3e      %.3e" % (n, k, i, fnk, alpha*tvnk, fn + alpha*tvn, pred_nkd))

            pred_positive = pred_nkd > 0.
            if ared_nkd >= sigma * pred_nkd and pred_positive:
              A.append_data(k, i, w, ared_nkd)
            elif pred_positive and A.compute_max()[0] < sigma * pred_nkd: #+ LnablaF*Drad:
              W.append_data(k+1, i)
            W.remove_data(k, i)
            xnk_temp[ind] = xn[ind]

          if not bool(W):
             break

        if Drad < 1:
            print("Trust region contracted for n = %d, k = %d." % (n, k))

        if len(A.data)==0:
            print("Set of Active patches is empty!")
            return xn

        # maxKey     = A.compute_max()[1]
        # ind        = patches.idx[maxKey[1]]
        # xn[ind]    = A.data[maxKey][0][0]
        txn = xn.copy()
        j0  = fn + alpha*tvn
        while len(A.data)!=0:
          maxKey     = A.compute_max()[1]
          temp_x     = A.data[maxKey][0][0]
          ind        = patches.idx[maxKey[1]] #should just pick out grid indices
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

    return xn



def lg_objective_var(x, lg_cm, di, f_vec):
    return lg_objective(x, lg_cm, di, f_vec)

def lg_jacobian_var(x, lg_cm, di, f_vec):
    return lg_jacobian(x, lg_cm, di, f_vec)

def main(N=2**12, alpha = 5e-5, numPatches=5, tol = 1e-6, usePlots = True, useParallel = False):
    # N = 2**14 #32768 #16384 #8192
    di = create_discretization_info(-1., 1., N, 5)

    lo_bangs = np.array([-1, 0, 1], dtype=np.int32)

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
    Delta0 = N // 16
    sigma = 1e-3
    maxiter = 100
    maxiter_k = 20
    h = 2./N

    # numPatches = 4
    bufferSize = int(np.floor(N/(numPatches*2)))
    patches = OneDPatches(di, numPatches, buffer=bufferSize)
    # == Optimization with convolution evaluated at Legendre-Gauss points ==
    opt_start = time.time()
    x_bs = blockslip(x, patches, lo_bangs, alpha, h, Delta0, sigma, maxiter, maxiter_k, tol, lg_cm, di, f_vec, useParallel = useParallel)
    timebs = time.time() - opt_start
    fbs = eval_f(x_bs)
    tvbs = eval_tv(x_bs)
    state_vec_bs = lg_cm.conv(x_bs)
    print("\n")

    ## would probably have to put below into slip
    # process = psutil.Process(os.getpid())
    # memoryUsage = process.memory_info().rss
    if np.log(N)/np.log(2) <= 16: #(memoryUsage*1e-9)>2: #check if memory usage is too high
      opt_start = time.time()
      x_s  = slip(eval_f, eval_jac, x, lo_bangs, alpha, h, Delta0, sigma, maxiter)
      time_s = time.time() - opt_start
      state_vec_s  = lg_cm.conv(x_s)
      print("N = ", N, "alpha = ", alpha, "Num Patch = ", numPatches, "||x_bs - x_s|| = ",       np.linalg.norm(x_bs - x_s), "||S(x_bs) - S(x_s)|| = ", np.linalg.norm(state_vec_bs - state_vec_s))
      fs = eval_f(x_s)
      tvs = eval_tv(x_s)
    else:
      print('Could not run slip')
      x_s = np.empty(x_bs.shape)
      x_s[:] = np.nan
      state_vec_s  = lg_cm.conv(x_s)
      fs = eval_f(x_s)
      tvs = eval_tv(x_s)
      time_s = np.nan

    if usePlots:
      plt.subplot(1, 2, 1)
      ind = np.linspace(0., 1., N)
      plt.plot(ind, x_bs)
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
      plt.xlabel('State') #add patches
      ax = plt.gca()
      ax.set_xlim([0., 1.])
      ax.set_ylim([-.015, 0.015])
      plt.tight_layout()

      if useParallel:
         plt.savefig("parallel_"+str(N)+"_"+str(alpha)+"_"+str(numPatches), format = 'eps', dpi=1200)
      else:
         plt.savefig("serial_"+str(N)+"_"+str(alpha)+"_"+str(numPatches), format = 'eps', dpi=1200)
      plt.close()
    return (fbs+alpha * tvbs, fs+alpha * tvs, fbs, tvbs, fs, tvs, timebs, time_s), (x_bs, state_vec_bs)

if __name__ == "__main__":
    main()



