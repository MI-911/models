import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
import numpy as np


class DeepQInterView(nn.Module):
    def __init__(self, alpha, n_entities, fc1_dims, fc2_dims):
        super(DeepQInterView, self).__init__()

        self.alpha = alpha
        self.n_entities = n_entities

        self.answer_layer = nn.Linear(n_entities * 2, fc1_dims)
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        self.question_layer = nn.Linear(fc2_dims, n_entities)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        self.device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = tt.from_numpy(observation).to(tt.float).to(self.device)
        x = tt.tanh(self.answer_layer(state))
        x = tt.tanh(self.fc1(x))
        x = tt.sigmoid(self.question_layer(x))

        return x


class InterviewAgent:
    def __init__(self, gamma, epsilon, alpha, n_movies, batch_size,
                 max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.question_space = [i for i in range(n_movies)]
        self.question_dim = n_movies
        self.answer_dim = self.question_dim * 2
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.Q_eval = DeepQInterView(alpha, n_entities=self.question_dim, fc1_dims=256, fc2_dims=256)

        self.Q_eval_prime = DeepQInterView(alpha, n_entities=self.question_dim, fc1_dims=256, fc2_dims=256)
        self.Q_eval_prime.load_state_dict(self.Q_eval.state_dict())
        self.Q_eval_prime.eval()

        self.state_memory = np.zeros((self.mem_size, self.answer_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.answer_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.question_dim), dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.update_counter = 0

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.question_dim)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - terminal
        self.mem_counter += 1

    def choose_action(self, observation, evaluation=False):
        rand = np.random.random()
        actions = self.Q_eval(observation.reshape((1, observation.shape[0])))
        if rand > self.EPSILON or evaluation:
            action = tt.argmax(actions).item()
        else:
            action = np.random.choice(self.question_space)
        return action

    def learn(self):
        if self.mem_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_counter if self.mem_counter < self.mem_size \
                else self.mem_size

            if self.update_counter % 150 == 0:
                print(f'Updating Q\'...')
                self.Q_eval_prime.load_state_dict(self.Q_eval.state_dict())

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.question_space, dtype=np.uint8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = tt.tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = tt.tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval_prime(state_batch).to(self.Q_eval.device).detach()
            q_next = self.Q_eval(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            target_update = reward_batch + \
                            self.GAMMA * tt.max(q_next, dim=1)[0] * terminal_batch
            for i in range(len(batch_index)):
                q_target[batch_index[i], action_indices[i]] = target_update[i]

            self.EPSILON = self.EPSILON * self.EPS_DEC if self.EPSILON > \
                                                          self.EPS_MIN else self.EPS_MIN

            loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
            loss.backward()

            # Clip the gradients
            for param in self.Q_eval.parameters():
                param.grad.data.clamp_(-1, 1)

            self.Q_eval.optimizer.step()
            self.update_counter += 1

            return loss

