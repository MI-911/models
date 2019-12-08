import matplotlib.pyplot as plt
import json
import os
import numpy as np
from utilities.util import plot_learning


MOVIES = 'MOVIES'
ENTITIES = 'ENTITIES'
BOTH = 'BOTH'

if __name__ == '__main__':
    f_names = ['1Q.json', '2Q.json', '3Q.json', '4Q.json', '5Q.json']

    for i, f_name in enumerate(f_names, start=1):
        with open(f'{f_name}') as fp:
            data = json.load(fp)

        train_ap = data['train']
        test_ap = data['test']

        plt.plot(train_ap, label='Training AP@20 at every epoch')
        plt.plot(test_ap, label='Test AP@20 at every epoch')
        plt.title(f'AP@20 for training and testing (50 epochs, {i} questions)')
        plt.xlabel('Epochs')
        plt.ylabel('AP@20')
        plt.show()

        running_ap = data['avg@50']
        running_eps = data['eps@50']

        x = [i for i in range(len(running_ap))]
        plot_learning(x, running_ap, running_eps)

        losses = data['losses']
        plt.plot(losses)
        plt.show()

        print(f'Train AP@20: {data["train"]} - ' + f'Test AP@20: {data["test"]}')
