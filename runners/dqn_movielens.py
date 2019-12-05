import gym
from models.dqn_movielens import DeepQInterView, InterviewAgent
from utilities.util import plot_learning
import numpy as np
from gym import wrappers
from data.cold_start import generate_movielens_dataset, MovieLensDataSet
from data.graph_loader import CollaborativeKnowledgeGraph
import torch as tt
import sys
import json


LIKED = 1
DISLIKED = -1


def convert_movielens_rating(r):
    if r > 3:
        return LIKED
    return DISLIKED


class Environment:
    def __init__(self, interview_length=3):
        self.data_set = generate_movielens_dataset(
            movielens_dir='../data/movielens/ratings.csv',
            top_n=100,
            rating_converter=convert_movielens_rating)

        # 1. Load the users
        self.train_users, self.test_users = self.data_set.split_users(split_ratio=0.75)
        self.train_users = MovieLensDataSet.shuffle(self.train_users)
        self.test_users = MovieLensDataSet.shuffle(self.test_users)

        # 2. Load the collaborative KG
        self.KG = CollaborativeKnowledgeGraph.load_from_users(
            user_set=self.train_users,
            liked_value=LIKED,
            disliked_value=DISLIKED
        )

        # 3. Set the current user
        self.current_user = None
        self.user_counter = 0
        self.n_train_users = len(self.train_users)
        self.n_test_users = len(self.test_users)

        # 4. Set the state
        self.n_movies = self.data_set.n_movies
        self.state_size = self.n_movies * 2  # One for each question/answer pair.
        self.state = np.zeros(self.state_size)

        # 5. Other info
        self.interview_length = interview_length
        self.current_interview_length = 0
        self.evaluation = False

    def choose_user(self, u_idx, train=True):
        # Manually choose a user and reset the state
        self.evaluation = not train
        self.current_user = self.train_users[u_idx] if train else self.test_users[u_idx]
        self.current_user.split()
        self.state = np.zeros(self.state_size)
        self.current_interview_length = 0

        return self.state

    def reset(self):
        # Choose a new user
        current_user_idx = self.user_counter % self.n_train_users
        self.current_user = self.train_users[current_user_idx]
        self.current_user.split()  # Initialise fresh answer sets
        self.user_counter += 1

        # Reset state
        self.state = np.zeros(self.state_size)
        self.current_interview_length = 0
        self.evaluation = False

        # Return the state as the observation
        return self.state

    def step(self, q):
        # Assuming an n_movies + n_entities vector, take the argmax as the question.

        # Ask the user what they think
        answer = self.current_user.ask(q, interviewing=True)
        self.current_interview_length += 1

        # Update the state, doubling the movie index
        q = q * 2
        self.state[q] = 1
        self.state[q + 1] = answer

        # Calculate a reward
        reward = self._calculate_reward(top_n=20)

        # Are we done?
        done = self.current_interview_length >= self.interview_length

        # Return the state as an observation
        return self.state, reward, done

    def _calculate_reward(self, top_n=20):
        # Perform PR with the liked entities as the source nodes
        # The reward is the average precision of the top-20.

        # 1. Extract the liked and disliked entities
        liked = np.where(self.state == LIKED)[0]
        disliked = np.where(self.state == DISLIKED)[0]
        # 1.1 The uneven indices are the ones containing ratings.
        #     Also, they are spread in doubled length to contain answers.
        #     For these items:
        #     [0, 1, 2, 3, 4, 5]
        #     We would have the item/answer pairs:
        #     [0, 0, 1, -1, 2, 1, 3, 1, 4, 1, 5, 0]
        #     So for every rating, find its doubled item index and
        #     halve it.
        liked = [(_i - 1) / 2 for _i in liked if not _i % 2 == 0]
        # disliked = [(_i - 1) / 2 for _i in disliked if not _i % 2 == 0]

        # 2. Pass to PPR, take the top 20
        top_n_liked = self.KG.ppr_top_n(liked, top_n=top_n)
        # top_n_disliked = self.KG.ppr_top_n(disliked, top_n=top_n)

        # 3. For the current user, use their evaluation set as the ground truth.
        #    For every movie in these predictions, see if they occur correctly in their sets.

        return self._average_precision(top_n_liked)

    def _average_precision(self, like_predictions):
        n = len(like_predictions)
        n_correct = 0
        pre_avg = 0
        for _i, prediction in enumerate(like_predictions, start=1):
            answer = self.current_user.ask(prediction, interviewing=False)
            if answer == 1:
                n_correct += 1
                pre_avg += n_correct / _i

        return pre_avg / n


def evaluate(num_users, train=True):
    scores = []
    # Test on all test users
    print(f'Evaluating on {num_users} test users...')
    with tt.no_grad():
        brain.Q_eval.eval()
        for u in range(num_users):
            interview_score = 0
            done = False
            observation = env.choose_user(u, train=train)

            while not done:
                action = brain.choose_action(observation, evaluation=True)
                observation_, reward, done = env.step(action)

                if done:
                    interview_score = reward

                observation = observation_

            scores.append(interview_score)

        print(f'Test avg. interview score: {np.mean(scores)}')
        return np.mean(scores)


if __name__ == '__main__':
    env = Environment(interview_length=0)
    for n_questions in [1, 2, 3, 4, 5]:
        env.interview_length = n_questions
        brain = InterviewAgent(gamma=0.99, batch_size=24,
                               n_movies=env.n_movies, alpha=0.003,
                               epsilon=1.0, eps_dec=0.999, eps_end=0.05)

        n_epochs = 50
        num_train_users = env.n_train_users
        num_test_users = env.n_test_users

        epsilon_history = []
        train_score_history = []
        test_score_history = []
        training_losses = []

        interview_scores = []
        avg_interview_scores = []

        n_interviews = 0

        evaluate_at_every = 50

        for epoch in range(n_epochs):
            train_score = 0
            test_score = 0

            MovieLensDataSet.shuffle(env.train_users)

            # Train on all train users
            print(f'Testing on {num_train_users} train users...')
            for u in range(num_train_users):
                brain.Q_eval.train()
                done = False
                interview_score = 0
                observation = env.reset()
                previous_questions = []
                transitions = []

                for i in range(n_questions):
                    state = env.state.copy()
                    action = brain.choose_action(observation)
                    answer = env.current_user.ask(action)
                    action_index = action * 2
                    env.state[action_index] = 1
                    env.state[action_index + 1] = answer
                    new_state = env.state.copy()
                    done = i == n_questions - 1
                    transitions.append([state, action, new_state, done])

                # Done, get a reward for the interview
                final_reward = env._calculate_reward(top_n=20)
                for state, action, new_state, done in transitions:
                    brain.store_transition(state, action, final_reward / n_questions, new_state, done)

                interview_scores.append(final_reward)

                #
                #
                #
                #
                #
                # while not done:
                #     action = brain.choose_action(observation)
                #     observation_, reward, done = env.step(action)
                #     if action in previous_questions:
                #         reward = 0  # Don't ask the same question again and again
                #     brain.store_transition(observation, action, reward, observation_,
                #                            done)
                #     observation = observation_
                #     previous_questions.append(action)
                #     if done:
                #         interview_score = reward
                loss = brain.learn()
                if loss is not None:
                    loss = loss.cpu().detach().numpy().sum()
                    training_losses.append(float(loss))

                interview_scores.append(interview_score)
                n_interviews += 1

                if u % evaluate_at_every == 0 and u > 0:
                    recent_avg_score = np.mean(interview_scores[-evaluate_at_every:])
                    avg_interview_scores.append(recent_avg_score)
                    epsilon_history.append(brain.EPSILON)

                    print(f'{n_interviews} interviews, Epsilon: {brain.EPSILON}, avg. interview score: {recent_avg_score}')

            # Only evaluate at the end of an epoch
            train_score_history.append(evaluate(num_train_users, train=True))
            test_score_history.append(evaluate(num_test_users, train=False))

        with open(f'../results/{n_epochs}_epochs_top_100/{n_questions}Q.json', 'w') as fp:
            json.dump({
                'losses': training_losses,
                'train': train_score_history,
                'test': test_score_history,
                'avg@50': avg_interview_scores,
                'eps@50': epsilon_history
            }, fp)

        # x = [i + 1 for i in range(len(avg_interview_scores))]
        # plot_learning(x, avg_interview_scores, epsilon_history)

    # plotLearning(x, test_score_history, epsilon_history)

#
# if __name__ == '__main__':
#     env = gym.make('LunarLander-v2')
#     brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
#                   input_dims=[8], alpha=0.003)
#
#     scores = []
#     eps_history = []
#     num_games = 500
#     score = 0
#     # uncomment the line below to record every episode.
#     #env = wrappers.Monitor(env, "tmp/space-invaders-1",
#     #video_callable=lambda episode_id: True, force=True)
#     for i in range(num_games):
#         if i % 10 == 0 and i > 0:
#             avg_score = np.mean(scores[max(0, i-10):(i+1)])
#             print('episode: ', i,'score: ', score,
#                  ' average score %.3f' % avg_score,
#                 'epsilon %.3f' % brain.EPSILON)
#         else:
#             print('episode: ', i,'score: ', score)
#         eps_history.append(brain.EPSILON)
#         done = False
#         observation = env.reset()
#         score = 0
#         while not done:
#             action = brain.chooseAction(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             brain.storeTransition(observation, action, reward, observation_,
#                                   done)
#             observation = observation_
#             brain.learn()
#
#         scores.append(score)
#
#     x = [i+1 for i in range(num_games)]
#     filename = str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
#                'Alpha' + str(brain.ALPHA) + 'Memory' + \
#                 str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) +'.png'
#     plotLearning(x, scores, eps_history, filename)