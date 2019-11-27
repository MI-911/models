from os.path import join
import requests


def download_mindreader(save_to='./data/mindreader'):
    """
    Downloads the mindreader dataset (ratings.csv and entities.csv) and
    stores the files in ./<save_to>/.
    :param save_to:
    """
    ratings_url = 'https://mindreader.tech/api/ratings'
    entities_url = 'https://mindreader.tech/api/entities'

    ratings_response = requests.get(ratings_url)
    entities_response = requests.get(entities_url)

    with open(join(save_to, 'ratings.csv'), 'wb') as rfp:
        rfp.write(ratings_response.content)
    with open(join(save_to, 'entities.csv'), 'wb') as efp:
        efp.write(entities_response.content)


if __name__ == '__main__':
    download_mindreader(save_to='../data/mindreader')
