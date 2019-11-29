import torch


def combine_movie_entity_index(data):
    combined = [{}, {}]

    index = 1
    for (u, m, rating, type) in data:
        if m not in combined[type]:
            combined[type][m] = index
            index += 1

    return combined


def batch_generator(data, batch_size):
    length = len(data)
    step = 0
    while True:
        cur_index = batch_size * step
        batch = data[cur_index: cur_index + batch_size]
        step += 1

        batch = list(zip(*batch))
        batch = torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.FloatTensor(batch[2])

        if cur_index + batch_size < length:
            yield batch
        else:
            return batch