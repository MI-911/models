import gc

import pandas as pd

ml_path = '../data/movielens'


def transform_imdb_id(imdb_id):
    return f'tt{str(imdb_id).zfill(7)}'


ratings = pd.read_csv(f'{ml_path}/ratings.csv')
links = pd.read_csv(f'{ml_path}/links.csv')
mapping = pd.read_csv(f'{ml_path}/mapping.csv')

# Merge ratings with links
ratings = ratings.merge(links, on='movieId')
ratings.imdbId = ratings.imdbId.map(lambda imdb_id: f'tt{str(imdb_id).zfill(7)}')

# Merge ratings with mapping
ratings = ratings.merge(mapping, on='imdbId')
ratings.dropna(inplace=True)

del mapping, links
gc.collect()
