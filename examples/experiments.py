import sys, getopt
import pandas as pd
import pickle
import numpy as np
import run_block_slip_algorithm
import resource


def main(exp):

    try:
      opts, args = getopt.getopt(exp, "e:", ["exp="])
    except getopt.GetoptError:
       print('usage: -e# or --exp=#')
       sys.exit()

    for opt, arg in opts:
       if opt in ("-e", "--exp"):
          exp = int(arg)
    print( "Running experiment: ", exp)

    if exp==1:
      N = [12,13,14,15,16] #[13, 14, 15, 16]
      NumPatch = [5,7,9,11]  # NumPatch = [5, 7, 9, 11]
      alpha = [2.5e-4,np.sqrt(5)*5e-5,5e-5,np.sqrt(5)*1e-5,1e-5] # [np.sqrt(5)*(10**-4), 10**-4, np.sqrt(5)**-1*(10**-4), np.sqrt(5)**(-2)*(10**-4)]
      midx = pd.MultiIndex.from_product([N, NumPatch, alpha])
      my_columns = [u'J(xbs)', u'J(xs)', u'f(xbs)', u'tv(xbs)', u'f(xs)', u'tv(xs)', u't(xbs)', u't(xs)']
      df = pd.DataFrame(index=midx, columns=my_columns)
      al_ind = 0
      for idx in midx:
          print(idx)
          al_ind += 1
          (results, variables) = run_block_slip_algorithm.main(N=2**idx[0], alpha = idx[2], numPatches=idx[1])
          print("\n\n")

          np.savetxt(str(idx[0])+str(idx[1])+str(np.mod(al_ind, 4))+'_control.csv', variables[0], delimiter=",")
          np.savetxt(str(idx[0])+str(idx[1])+str(np.mod(al_ind, 4))+'_state.csv', variables[1], delimiter=",")
          for i in range(0, len(results)):
              df.loc[idx, my_columns[i]] = results[i]
      df.to_pickle('exp1')
    elif exp==2:
      N = [13]
      NumPatch = [5,]
      alpha = [1e-4]

      midx = pd.MultiIndex.from_product([N, NumPatch, alpha])
      my_columns = [u'f(xbs)', u'tv(xbs)', u'f(xs)', u'tv(xs)', u't(xbs)', u't(xs)']
      df = pd.DataFrame(index=midx, columns=my_columns)

      for idx in midx:
          print(idx)
          results = run_block_slip_algorithm.main(N=2**idx[0], alpha = idx[2], numPatches=idx[1])
          for i in range(0, len(results)):
              df.loc[idx, my_columns[i]] = results[i]
      df.to_pickle('exp2')


    print(df.to_latex())

if __name__ == "__main__":
    main(sys.argv[1:])
