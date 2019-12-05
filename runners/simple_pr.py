from data.graph_loader import CollaborativeKnowledgeGraph
from data.cold_start import generate_movielens_dataset
import pandas as pd
import networkx as nx
import numpy as np
import random


def rating_convert(r):
    if r > 3:
        return 1
    return -1


def average_precision(user, predictions):
    n = len(predictions)
    n_correct = 0
    pre_avg = 0

    for i, prediction in enumerate(predictions, start=1):
        answer = user.ask(prediction, interviewing=False)
        if answer == 1:
            n_correct += 1
            pre_avg += n_correct / i

    return pre_avg / n


if __name__ == '__main__':
    random.seed(42)
    data_set = generate_movielens_dataset(
        movielens_dir=f'../data/movielens/ratings.csv',
        top_n=100,
        rating_converter=rating_convert)

    train_users, test_users = data_set.split_users()
    _ = [u.split() for u in train_users]
    _ = [u.split() for u in test_users]
    KG = CollaborativeKnowledgeGraph.load_from_users(train_users)

    ranked_movies = KG.ppr_top_n([], top_n=20)

    aps = []

    for u in train_users:
        ap = average_precision(u, ranked_movies[:20])
        print(f'AP@20: {ap}')
        aps.append(ap)

    print(f'MAP@20: {sum(aps) / len(aps)}')

    input()
    aps = []

    for i, u in enumerate(train_users):
        ratings = [m for m, r in u.movie_interview_answers.items() if r == 1]
        if len(ratings) <= 3:
            continue

        seeds = np.random.choice(ratings, 3)
        ranked_movies = KG.ppr_top_n(seeds, top_n=20)
        ap = average_precision(u, ranked_movies[:20])
        print(f'AP@20: {ap} ({(i / len(train_users)) * 100 : 2.2f}%)')
        aps.append(ap)

    print(f'MAP@20: {sum(aps) / len(aps)}')





