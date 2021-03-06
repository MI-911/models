import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from json import load

if __name__ ==  '__main__':
    sns.kdeplot(load(open('./results/movielens_pairwise_distance.json', 'r'))['mean_path_lengths'])
    plt.show()

