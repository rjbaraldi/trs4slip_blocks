import numpy as np
import pandas as pd

import run_block_slip_algorithm


def main():
    N = [12, 13, 14, 15]
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

if __name__ == "__main__":
    main()
