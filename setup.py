from data.download_data import download_mindreader, preprocess_user_major, download_graph

if __name__ == '__main__':
    download_mindreader(save_to='./data/mindreader', only_completed=True)
    # download_graph(save_to='./data/graph')

    with open('./data/mindreader/user_ratings_map.json', 'w') as fp:
        preprocess_user_major(from_path='./data/mindreader', fp=fp)
