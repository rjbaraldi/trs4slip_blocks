import sys, getopt
import pandas as pd

import run_block_slip_algorithm


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
      N = [12, 13, 14, 15, 16]
      NumPatch = [7, 15]
      alpha = [2.5e-4, 1e-4, 5e-5, 1e-5]

      midx = pd.MultiIndex.from_product([N, NumPatch, alpha])
      my_columns = [u'f(xbs)', u'tv(xbs)', u'f(xs)', u'tv(xs)', u't(xbs)', u't(xs)']
      df = pd.DataFrame(index=midx, columns=my_columns)

      for idx in midx:
          print(idx)
          results = run_block_slip_algorithm.main(N=2**idx[0], alpha = idx[2], numPatches=idx[1])
          for i in range(0, len(results)):
              df.loc[idx, my_columns[i]] = results[i]
      df.to_pickle('exp')
    elif exp==2:
      N = [13, 14, 15]
      NumPatch = [3, 5, 7, 9, 11, 13]
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
