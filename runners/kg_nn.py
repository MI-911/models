import torch as tt
import torch.nn as nn
import torch.optim as optim

from data.cold_start import generate_dataset, DataSet, LIKE, DISLIKE, UNKNOWN
from models.all_nn import InterviewingNeuralNetwork
from data.graph_loader import load_graph

import numpy as np
from matplotlib import pyplot as plt

import random
import json
from os.path import join

MOVIES = 'MOVIES'
ENTITIES = 'ENTITIES'
BOTH = 'BOTH'


def answer_vector(item, answer, answer_vector_dim=400):
    a = tt.zeros(answer_vector_dim)
    item_answer_idx = item * 2
    a[item_answer_idx] = 1
    a[item_answer_idx+1] = answer
    return a


def ratings_vector(user, asking_for, n_items=200):
    ratings = tt.zeros(n_items)
    if asking_for == MOVIES:
        for o, r in user.m_test_answers.items():
            ratings[o] = r
        return ratings
    if asking_for == ENTITIES:
        for o, r in user.e_test_answers.items():
            ratings[o] = r
    if asking_for == BOTH:
        for o, r in user.m_test_answers.items():
            ratings[o] = r
        for o, r in user.e_test_answers.items():
            ratings[o] = r

    return ratings


def ask(user, item, asking_for):
    answer = UNKNOWN
    if asking_for == MOVIES:
        answer = user.ask_movie(item)
    if asking_for == ENTITIES:
        answer = user.ask_entity(item)
    if asking_for == BOTH:
        answer = user.ask_movie(item)
        if answer == UNKNOWN:
            answer = user.ask_entity(item)

    return answer_vector(item, answer, answer_vector_dim=answer_vector_dim)


def run_eval(model, users, evaluation=False, M=10):
    average_precisions = []

    with tt.no_grad():
        model.eval()
        eval_loss = tt.tensor(0.0)

        for i, u in enumerate(users):

            # Ask Q questions, saving the answers each time
            e_answers = tt.zeros(answer_vector_dim)
            for q in range(n_questions):
                e_question = model(e_answers, interviewing=True)
                e_question = e_question.argmax().numpy().sum()  # Extract the actual index
                e_answer = ask(u, e_question, asking_for=asking_for)
                e_answers = e_answers + e_answer  # Add this answer to the history

            # Generate rating predictions, calculate loss
            e_predicted_ratings = model(e_answers, interviewing=False)
            e_ratings = ratings_vector(u, asking_for=MOVIES, n_items=n_items)

            eval_loss += loss_fn(e_predicted_ratings, e_ratings)

            # Calculate average AP@M
            n_correct = 0
            precisions = []

            # Consider only movie ratings
            e_predicted_ratings = e_predicted_ratings[:data_set.n_movies]
            e_ratings = e_ratings[:data_set.n_movies]

            sorted_predicted_ratings = list(sorted([(r_i, r) for r_i, r in enumerate(e_predicted_ratings)], reverse=True))
            threshold = np.median(e_predicted_ratings)

            for m in range(1, M + 1, 1):
                item_index, prediction = sorted_predicted_ratings[m]
                if prediction > threshold:
                    if e_ratings[item_index] == LIKE:
                        n_correct += 1
                        precisions.append(n_correct / m)

            average_precisions.append(np.mean(precisions) if len(precisions) > 0 else 0)

    return eval_loss / len(users), np.mean(average_precisions)


def kg_loss(answer_vector):
    # Find all the entities that have been liked and disliked, respectively
    liked =


if __name__ == '__main__':
    random.seed(42)

    min_num_ratings = 5
    top_n = None

    data_set = generate_dataset(
        mindreader_dir='../data/mindreader',
        top_n=top_n,
    )

    # Define hyper parameters
    n_epochs = 100
    learning_rate = 0.001
    n_questions = 2
    batch_size = 32
    interview_layer_size = 512
    n_items = data_set.n_movies + data_set.n_entities
    answer_vector_dim = n_items * 2
    question_vector_dim = n_items

    print(f'Training on {data_set.n_movies} movies and {data_set.n_entities} entities.')

    # Build model
    model = InterviewingNeuralNetwork(n_items=n_items, hidden_size=interview_layer_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate user sets and shuffle their answer sets
    train_users, test_users = data_set.split_users()
    train_users = DataSet.shuffle(train_users)
    test_users = DataSet.shuffle(test_users)

    n_users = len(train_users)
    print(f'Loaded {len(train_users)} training users and {len(test_users)} test users')

    # What are we asking for?
    asking_for = MOVIES

    # Training histories
    train_history = []
    test_history = []
    train_precision = []
    test_precision = []
    M = 10  # MAP@M

    # Save dir
    SAVE_DIR = '../results/'

    for e in range(n_epochs):
        # Shuffle the data
        # train_users = DataSet.shuffle(train_users)  # Run this to shuffle users and their answers
        random.shuffle(train_users)  # Run this to shuffle the order of the users, only
        # test_users = DataSet.shuffle(test_users)

        # Activate gradients
        model.train()
        batch_loss = tt.tensor(0.0)

        for i, u in enumerate(train_users):

            # Ask Q questions, saving the answers each time
            answers = tt.zeros(answer_vector_dim)
            for q in range(n_questions):
                # TODO: If the model could not answer the first question, it will
                #       try asking that same question over and over. This is bad.
                #       Make the 200-d vector into a 400-d vector to retain question/answer
                #       pairs.
                question = model(answers, interviewing=True)
                question = question.argmax().numpy().sum()  # Extract the actual index
                answer = ask(u, question, asking_for=asking_for)
                answers = answers + answer  # Add this answer to the history

            # Generate rating predictions, calculate loss
            predicted_ratings = model(answers, interviewing=False)
            ratings = ratings_vector(u, n_items=n_items, asking_for=MOVIES)

            loss = loss_fn(predicted_ratings, ratings)

            batch_loss += loss

            if i % batch_size == 0 and i > 0:
                # Calculate gradients and backprop
                batch_loss.backward()
                optimizer.step()
                batch_loss = tt.tensor(0.0)

                # Zero out the gradients
                model.zero_grad()

        # Run post-epoch evaluation
        train_loss, train_map = run_eval(model, train_users, M=M)
        test_loss, test_map = run_eval(model, test_users, M=M)

        print(f'Epoch {e}')
        print(f'    Train loss:   {train_loss}')
        print(f'    Test loss:    {test_loss}')
        print(f'    Train MAP@10: {train_map}')
        print(f'    Test MAP@10:  {test_map}')

        train_history.append(train_loss)
        test_history.append(test_loss)
        train_precision.append(train_map)
        test_precision.append(test_map)

    # Post training stats
    stats = {
        'n_epochs': n_epochs,
        'asking_for': asking_for.lower(),
        'min_mse': {
            'train': float(min(train_history).numpy().sum()),
            'test': float(min(test_history).numpy().sum())
        },
        f'max_map@{M}': {
            'train': max(train_precision),
            'test': max(test_precision)
        }
    }

    f_name = f'INN-{asking_for}-{n_epochs}-{n_questions}Q'

    with open(join(SAVE_DIR, f'{f_name}.json'), 'w') as fp:
        json.dump(stats, fp, indent=True)

    plt.plot(train_history, color='orange', label='Train MSE')
    plt.plot(test_history, color='skyblue', label='Test MSE')
    plt.legend()
    plt.title(f'Interviewing Neural Network ({n_epochs} epochs), asking for {asking_for.lower()} (lr={learning_rate})')
    plt.savefig(join(SAVE_DIR, f'{f_name}-MSE.png'))
    plt.clf()

    plt.plot(train_precision, color='orange', label='Train MAP@10')
    plt.plot(test_precision, color='skyblue', label='Test MAP@10')
    plt.legend()
    plt.title(f'Interviewing Neural Network ({n_epochs} epochs), asking for {asking_for.lower()} (lr={learning_rate})')
    plt.savefig(join(SAVE_DIR, f'{f_name}-MAP.png'))
    plt.clf()



