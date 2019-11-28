import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from json import load

if __name__ ==  '__main__':
    sns.kdeplot(load(open('./results/mindreader_pairwise_distance_liked.json', 'r'))['mean_path_lengths'])
    plt.title('Pairwise like-like path length (mean per user)')
    plt.show()

