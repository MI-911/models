from os.path import join
import requests


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

    ratings_response = requests.get(ratings_url)
    entities_response = requests.get(entities_url)

    with open(join(save_to, 'ratings.csv'), 'wb') as rfp:
        rfp.write(ratings_response.content)
    with open(join(save_to, 'entities.csv'), 'wb') as efp:
        efp.write(entities_response.content)


if __name__ == '__main__':
    download_mindreader(save_to='../data/mindreader', only_completed=True)
