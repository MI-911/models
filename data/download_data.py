from os.path import join
import requests
import pandas as pd
import json
import os, io


def download_mindreader(save_to='./data/mindreader', only_completed=False):
    """
    Downloads the mindreader dataset.
    :param save_to: Directory to save ratings.csv and entities.csv.
    :param only_completed: If True, only downloads ratings for users who reached the final screen.
    """
    ratings_url = 'https://mindreader.tech/api/ratings'
    entities_url = 'https://mindreader.tech/api/entities'

    if only_completed:
        ratings_url += '?final=yes'

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    ratings_response = requests.get(ratings_url)
    entities_response = requests.get(entities_url)

    ratings = pd.read_csv(io.BytesIO(ratings_response.content))
    entities = pd.read_csv(io.BytesIO(entities_response.content))

    with open(join(save_to, 'ratings.csv'), 'w') as rfp:
        pd.DataFrame.to_csv(ratings, rfp)
    with open(join(save_to, 'entities.csv'), 'w') as efp:
        pd.DataFrame.to_csv(entities, efp)

    ratings = [(uid, uri, rating) for uid, uri, rating in ratings[['userId', 'uri', 'sentiment']].values]
    entities = [(uri, name, labels) for uri, name, labels in entities[['uri', 'name', 'labels']].values]

    # Filter out rating entities that aren't present in the entity set
    e_uris = [uri for uri, name, labels in entities]
    ratings = [(uid, uri, rating) for uid, uri, rating in ratings if uri in e_uris]

    with open(join(save_to, 'ratings_clean.json'), 'w') as rfp:
        json.dump(ratings, rfp)
    with open(join(save_to, 'entities_clean.json'), 'w') as efp:
        json.dump(entities, efp)


def preprocess_user_major(from_path='./data/mindreader', fp=None):
    """
    Stores ratings in a user-major fashion, e.g.:
    {
        user_1 = {
            'movies' : [(movie_1, 1), (movie_2, -1), (movie_3, 0)],
            'entities' : [(entity_1, 1), (entity_2, -1), (entity_3, 0)]
        },

        ...
    }
    :param from_path: Where to load ratings.csv and entities.csv from.
    :param fp: If not none, makes a JSON dump of the ratings to this file pointer.
    :return: The ratings map as a dictionary.
    """
    ratings_path = join(from_path, 'ratings_clean.json')
    entities_path = join(from_path, 'entities_clean.json')

    with open(ratings_path) as rfp:
        ratings = json.load(rfp)
    with open(entities_path) as efp:
        entities = json.load(efp)

    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    u_r_map = {}
    for uid, uri, rating in ratings:
        if uid not in u_r_map:
            u_r_map[uid] = {'movies': [], 'entities': []}
        if 'Movie' in label_map[uri]:
            u_r_map[uid]['movies'].append((uri, rating))
        else:
            u_r_map[uid]['entities'].append((uri, rating))

    if fp:
        json.dump(u_r_map, fp, indent=True)

    return u_r_map


if __name__ == '__main__':
    download_mindreader(save_to='../data/mindreader', only_completed=True)
    with open('../data/mindreader/user_ratings_map.json', 'w') as fp:
        preprocess_user_major(from_path='../data/mindreader', fp=fp)
