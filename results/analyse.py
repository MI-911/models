import matplotlib.pyplot as plt
import json
import os
import numpy as np

MOVIES = 'MOVIES'
ENTITIES = 'ENTITIES'
BOTH = 'BOTH'


def load_files():
    json_stats = []
    for file_name in os.listdir('./'):
        if file_name == 'analyse.py':
            continue

        with open(file_name) as fp:
            print(f'Loading {file_name}...')
            json_stats.append(json.load(fp))

    return json_stats


if __name__ == '__main__':
    json_stats = load_files()

    movies_stats = [stat for stat in json_stats if stat['asking_for'] == MOVIES.lower()]
    entities_stats = [stat for stat in json_stats if stat['asking_for'] == ENTITIES.lower()]
    both_stats = [stat for stat in json_stats if stat['asking_for'] == BOTH.lower()]

    movie_test_mses = [stat['train']['mse_history'] for stat in movies_stats]
    entity_test_mses = [stat['train']['mse_history'] for stat in entities_stats]
    both_test_mses = [stat['train']['mse_history'] for stat in both_stats]

    # TODO: Plot the minimum MSE we achieved with each question category
    #       at different interview lengths
    movie_performances = [min(mses) for mses in movie_test_mses]
    entity_performances = [min(mses) for mses in entity_test_mses]
    both_performances = [min(mses) for mses in both_test_mses]

    n_groups = 11

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    rects1 = plt.bar(index + bar_width * 0, movie_performances, bar_width, alpha=opacity, color='y', label='Movies')
    rects2 = plt.bar(index + bar_width * 1, entity_performances, bar_width, alpha=opacity, color='b', label='Entities')
    rects2 = plt.bar(index + bar_width * 2, both_performances, bar_width, alpha=opacity, color='g', label='Both')

    plt.xlabel('N. questions asked')
    plt.ylabel('Minimum test MSE during training')
    plt.title('Minimum MSEs attained with interviews of varying length')
    plt.xticks(index + bar_width, (q for q in range(n_groups)))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # TODO: Plot how the MSE changed during training for different interview
    #       lengths, separately for each question category

    for cat, mses_set in [(MOVIES, movie_test_mses), (ENTITIES, entity_test_mses), (BOTH, both_test_mses)]:
        [plt.plot(mses, label=f'{q} questions') for q, mses in enumerate(mses_set)]
        plt.xlabel('Batch updates')
        plt.ylabel('Test MSE')
        plt.title(f'MSE progression for {cat.lower()}-interviews of varying lengths')

        plt.legend()
        plt.show()






