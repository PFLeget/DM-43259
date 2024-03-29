import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import pickle
import copy
import time
import pandas as pd
from interpolate_over import InterpolateOverDefectGaussianProcess

def main():

    dic = pickle.load(open("out_test_0.pkl", "rb"))
    maskedImage = dic['in']['maskedImage']

    solvers = ['treegp', 'george', 'gpytorch']
    solvers_method = ['cholesky', 'HODLR', 'GPExact']
    methods = ['block', 'spanset']

    # Create an empty dataframe to store the results
    df = pd.DataFrame(columns=['Solver', 'Method', 'Time'])

    for i, s in enumerate(solvers):
        for m in methods:
            print(f'{s} | {solvers_method[i]} | {m}')
            to_interpolate = copy.deepcopy(maskedImage)
            GP = InterpolateOverDefectGaussianProcess(to_interpolate, defects=["SAT"],
                                                    fwhm=5, block_size=40, solver=s,
                                                    method=m)

            # Perform the interpolation and record the time for each solver and method
            start_time = time.time()
            GP.interpolate_over_defects()
            end_time = time.time()
            execution_time = end_time - start_time

            # Append the results to the dataframe
            df = pd.concat([df, pd.DataFrame({'Solver': [s], 'Method': [m], 'Time': [execution_time]})], ignore_index=True)

            plt.figure(figsize=(12, 12))
            plt.imshow(GP.maskedImage.getImage().array, vmin=400, vmax=800, cmap=plt.cm.Greys_r)
            plt.gca().invert_yaxis()
            plt.title(f'{s} | {solvers_method[i]} | {m}')
            plt.savefig(f'{s}_{solvers_method[i]}_{m}.png')
            plt.close()
    print(df)

if __name__ == "__main__":
    main()
