import gym
from models.dqn import DeepQInterView, InterviewAgent
from utilities.util import plotLearning
import numpy as np
from gym import wrappers
from data.cold_start import generate_dataset, DataSet
from data.graph_loader import load_labelled_graph


class Environment:
    def __init__(self, interview_length=3):
        self.data_set = generate_dataset(
            mindreader_dir='../data/mindreader',
            top_n=100,
        )

        # 1. Load the users
        self.train_users, self.test_users = self.data_set.split_users(split_ratio=0.75)
        self.train_users = DataSet.shuffle(self.train_users)
        self.test_users = DataSet.shuffle(self.test_users)

        # 2. Load the KG

        # 3. Set the current user
        self.current_user = None
        self.user_counter = 0
        self.n_train_users = len(self.train_users)

        # 4. Set the state
        self.n_movies = self.data_set.n_movies
        self.n_entities = self.data_set.n_entities
        self.state_size = (self.n_movies + self.n_entities) * 2  # One for each question/answer pair.
        self.state = np.zeros(self.state_size)

        # 5. Other info
        self.interview_length = interview_length
        self.current_interview_length = 0

    def reset(self):
        # Choose a new user
        current_user_idx = self.user_counter % self.n_train_users
        self.current_user = self.train_users[current_user_idx]
        self.user_counter += 1

        # Reset state
        self.state = np.zeros(self.state_size)
        self.current_interview_length = 0

        # Return the state as the observation
        return self.state

    def step(self, q):
        # Assuming an n_movies + n_entities vector, take the argmax as the question.

        # Ask the user what they think
        answer = self._ask(q)
        self.current_interview_length += 1

        # Update the state
        self.state[q] = 1
        self.state[q + 1] = answer

        # Calculate a reward
        # TODO: This loss must come from the KG!
        reward = self.state.sum()

        # Are we done?
        done = self.current_interview_length >= self.interview_length

        # Return the state as an observation
        return self.state, reward, done

    def _ask(self, question):
        answer = self.current_user.ask_movie(question)
        if answer == 0:
            answer = self.current_user.ask_entity(question)

        return answer

    def _calculate_reward(self):
        # Perform PR with the liked entities as the source nodes
        # The reward is the average precision of the top-20.33
        pass


if __name__ == '__main__':
    env = Environment()
    brain = InterviewAgent(gamma=0.99, batch_size=64,
                           n_movies=env.n_movies, n_entities=env.n_entities, alpha=0.003,
                           epsilon=1.0, eps_dec=0.999, eps_end=0.1)

    scores = []
    eps_history = []
    num_games = 50000
    score = 0

    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode: ', i, 'score: ', score)
        eps_history.append(brain.EPSILON)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = brain.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            brain.store_transition(observation, action, reward, observation_,
                                  done)
            observation = observation_
            brain.learn()

        scores.append(score)

    x = [i + 1 for i in range(num_games)]
    plotLearning(x, scores, eps_history)

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