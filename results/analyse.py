import matplotlib.pyplot as plt
import matplotlib
import json
import os
import numpy as np


def chart_cofactor_fmf_results():
    # First, show the Matrix Factorization data
    with open('../results/matrix_factorization/matrix_factorization_25_iter_25_k.json') as fp:
        mf = json.load(fp)
    with open('../results/cofactor/cofactor_25_iter_25_k.json') as fp:
        cofactor = json.load(fp)

    plt.plot(mf['training']['map'], marker='o', color='g', label='MF (training)')
    plt.plot(mf['testing']['map'], marker='+', color='g', label='MF (test)')

    plt.plot(cofactor['training']['map'], marker='o', color='b', label='CoFactor (training)')
    plt.plot(cofactor['testing']['map'], marker='+', color='b', label='CoFactor (testing)')

    plt.title('Comparison between vanilla MF and Explicit CoFactor on MovieLens 100K')
    plt.xlabel('ALS updates')
    plt.ylabel('MAP@20')

    plt.legend()

    plt.show()

    # Then, show the same for FMF and FMF w. CoFactor
    with open('../results/functional_matrix_factorization/fmf_broad_updates_10_iter_25_k.json') as fp:
        fmf = json.load(fp)
    with open('../results/functional_matrix_factorization/fmf_cofactor_broad_updates_10_iter_25_k.json') as fp:
        fmf_cofactor = json.load(fp)

    plt.plot(fmf['training']['map'], marker='o', color='g', label='FMF (training)')
    plt.plot(fmf['testing']['map'], marker='+', color='g', label='FMF (test)')

    plt.plot(fmf_cofactor['training']['map'], marker='o', color='b', label='Joint FMF (training)')
    plt.plot(fmf_cofactor['testing']['map'], marker='+', color='b', label='Joint FMF (test)')

    plt.title('Comparison between vanilla FMF and Joint FMF on MovieLens 100K')
    plt.xlabel('ALS iterations')
    plt.ylabel('MAP@20')

    plt.legend()

    plt.show()

    # Then, show how MAP@20 and RMSE changes when we switch from broad to narrow updates
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}

    matplotlib.rc('font', **font)
    with open('../results/functional_matrix_factorization/fmf_10_iter_25_k.json') as fp:
        fmf_narrow = json.load(fp)
    with open('../results/functional_matrix_factorization/fmf_cofactor_10_iter_25_k.json') as fp:
        fmf_cofactor_narrow = json.load(fp)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    ax2 = ax1.twinx()

    # # Broad FMF
    line, = ax2.plot(fmf['training']['rmse'], color='b', label='(B)FMF (RMSE, training)', linewidth=2, markersize=10)
    line.set_dashes([10, 10, 10, 10])
    line, = ax2.plot(fmf['testing']['rmse'], color='b', label='(B)FMF (RMSE, test)', linewidth=2, markersize=10)
    line.set_dashes([10, 10, 10, 10])
    ax1.plot(fmf['training']['map'], marker='o', color='b', label='(B)FMF (MAP@20, training)', linewidth=2, markersize=10)
    ax1.plot(fmf['testing']['map'], marker='+', color='b', label='(B)FMF (MAP@20, test)', linewidth=2, markersize=10)

    # Broad CoFactor
    # line, = ax2.plot(fmf_cofactor['training']['rmse'], color='b', label='Joint (B)FMF (RMSE, training)', linewidth=2, markersize=10)
    # line.set_dashes([10, 10, 10, 10])
    # line, = ax2.plot(fmf_cofactor['testing']['rmse'], color='b', label='Joint (B)FMF (RMSE, test)', linewidth=2, markersize=10)
    # line.set_dashes([5, 5, 5, 5])
    # ax1.plot(fmf_cofactor['training']['map'], marker='o', color='b', label='Joint (B)FMF (MAP@20, training)', linewidth=2, markersize=10)
    # ax1.plot(fmf_cofactor['testing']['map'], marker='+', color='b', label='Joint (B)FMF (MAP@20, test)', linewidth=2, markersize=10)

    # # Narrow FMF
    line, = ax2.plot(fmf_narrow['training']['rmse'], color='orange', label='(N)FMF (RMSE, training)', linewidth=2, markersize=10)
    line.set_dashes([10, 10, 10, 10])
    line, = ax2.plot(fmf_narrow['testing']['rmse'], color='orange', label='(N)FMF (RMSE, test)', linewidth=2, markersize=10)
    line.set_dashes([10, 10, 10, 10])
    ax1.plot(fmf_narrow['training']['map'], marker='o', color='orange', label='(N)FMF (MAP@20, training)', linewidth=2, markersize=10)
    ax1.plot(fmf_narrow['testing']['map'], marker='+', color='orange', label='(N)FMF (MAP@20, test)', linewidth=2, markersize=10)

    # Narrow CoFactor
    # line, = ax2.plot(fmf_cofactor_narrow['training']['rmse'], color='orange', label='Joint (N)FMF (RMSE, training)', linewidth=2, markersize=10)
    # line.set_dashes([10, 10, 10, 10])
    # line, = ax2.plot(fmf_cofactor_narrow['testing']['rmse'], color='orange', label='Joint (N)FMF (RMSE, test)', linewidth=2, markersize=10)
    # line.set_dashes([5, 5, 5, 5])
    # ax1.plot(fmf_cofactor_narrow['training']['map'], marker='o', color='orange', label='Joint (N)FMF (MAP@20, training)', linewidth=2, markersize=10)
    # ax1.plot(fmf_cofactor_narrow['testing']['map'], marker='+', color='orange', label='Joint (N)FMF (MAP@20, test)', linewidth=2, markersize=10)

    plt.title('RMSE/MAP under broad/narrow embedding updates', fontsize=24)
    plt.xlabel('ALS iterations')
    ax1.set_ylabel('MAP@20', color='b', fontsize=24)
    ax2.set_ylabel('RMSE', color='orange', fontsize=24)
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    plt.show()


if __name__ == '__main__':
    chart_cofactor_fmf_results()