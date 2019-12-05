import numpy as np
import math

LIKE = 1
DISLIKE = -1
UNKNOWN = 0


class Node:
    def __init__(self, users, fmf):
        self.users = users
        self.fmf = fmf  # So we can use the item embeddings
        self.profile = None
        self.like_child = None
        self.dislike_child = None
        self.unknown_child = None
        self.p = None  # The split item
        self.depth = None

    def calculate_profile(self):
        coeff_matrix = np.zeros((self.fmf.k, self.fmf.k))
        coeff_vector = np.zeros(self.fmf.k)

        for user in self.users:
            for m, r in user.movie_answers.items():
                m_embedding = self.fmf.M[m].reshape((1, self.fmf.k))
                coeff_matrix += m_embedding.T @ m_embedding + np.eye(self.fmf.k) * 0.0015
                coeff_vector += m_embedding.reshape(self.fmf.k) * r

        try:
            coeff_matrix = np.linalg.inv(coeff_matrix)
            self.profile = coeff_matrix.dot(coeff_vector)
        except Exception as e:
            print(e)
            print(self.users)
            print(coeff_matrix)

        # Calculate and return loss
        sse = 0
        for user in self.users:
            for m, r in user.movie_answers.items():
                prediction = self.profile @ self.fmf.M[m]
                sse += (prediction - r) ** 2

        return sse

    def split_users(self, o, is_entity):
        RL, RD, RU = [], [], []
        for user in self.users:
            r = user.ask(o, is_entity)
            if r == LIKE:
                RL.append(user)
            elif r == DISLIKE:
                RD.append(user)
            else:
                RU.append(user)

        return RL, RD, RU

    def split(self, is_entity, depth):
        self.depth = depth
        smallest_loss = math.inf
        best_split_item = None
        bestRLn, bestRDn, bestRUn = None, None, None

        for p in self.fmf.question_items:
            RL, RD, RU = self.split_users(p, is_entity)
            RLn, RDn, RUn = None, None, None  # Assign these now, it's OK if one or two is None later

            loss = 0

            # Calculate optimal profiles for each group
            # TODO: Consider introducing a minimum group size
            min_group_size = 5
            if len(RL) > min_group_size:
                RLn = Node(RL, self.fmf)
                loss += RLn.calculate_profile()
            if len(RD) > min_group_size:
                RDn = Node(RD, self.fmf)
                loss += RDn.calculate_profile()
            if len(RU) > min_group_size:
                RUn = Node(RU, self.fmf)
                loss += RUn.calculate_profile()

            if loss < smallest_loss:
                smallest_loss = loss
                bestRLn = RLn
                bestRDn = RDn
                bestRUn = RUn
                best_split_item = p

        # Set the child nodes
        self.like_child = bestRLn
        self.dislike_child = bestRDn
        self.unknown_child = bestRUn

        self.p = best_split_item
        self.fmf.question_items.remove(self.p)

        # Split the child nodes
        if depth < self.fmf.max_depth:
            if self.like_child is not None:
                self.like_child.split(is_entity, depth + 1)
            if self.dislike_child is not None:
                self.dislike_child.split(is_entity, depth + 1)
            if self.unknown_child is not None:
                self.unknown_child.split(is_entity, depth + 1)

    def interview(self, user, is_entity):
        r = user.ask(self.p, is_entity)
        if r == LIKE:
            if self.like_child is not None:
                return self.like_child.interview(user, is_entity)
        elif r == DISLIKE:
            if self.dislike_child is not None:
                return self.dislike_child.interview(user, is_entity)
        elif r == UNKNOWN:
            if self.unknown_child is not None:
                return self.unknown_child.interview(user, is_entity)

        return self.profile

    def print(self, idx_name_map):
        if self.p is None:
            return
        if self.like_child is not None:
            self.like_child.print(idx_name_map)
        if self.dislike_child is not None:
            self.dislike_child.print(idx_name_map)
        if self.unknown_child is not None:
            self.unknown_child.print(idx_name_map)


class FunctionalMatrixFactorizaton:
    def __init__(self, n_users, n_movies, n_entities, k, max_depth, entities_in_question_set=False, m_idx_name_map=None, e_idx_name_map=None):
        self.n_movies = n_movies
        self.n_users = n_users
        self.n_entities = n_entities
        self.k = k
        self.max_depth = max_depth

        self.m_idx_name_map = m_idx_name_map
        self.e_idx_name_map = e_idx_name_map

        # Movie embeddings
        self.M = np.random.rand(n_movies, k)
        self.U = np.random.rand(n_users, k)

        # Questionable entities
        self.is_entity = entities_in_question_set

    def update_user_embeddings(self, train_users, test_users, tree, is_entity):
        for user in train_users:
            self.U[user.id] = tree.interview(user, is_entity)
        for user in test_users:
            self.U[user.id] = tree.interview(user, is_entity)

    def update_item_embeddings(self, train_users):
        for m in range(self.n_movies):
            coeff_matrix = np.zeros((self.k, self.k))
            coeff_vector = np.zeros(self.k)

            # Find the users who have seen this movie
            users = [user for user in train_users if not user.ask(m, False) == UNKNOWN]
            for user in users:
                r = user.ask(m, False)
                Ta = self.U[user.id].reshape((1, self.k))

                coeff_matrix += Ta.T @ Ta + np.eye(self.k) * 0.0015
                coeff_vector += r * Ta.reshape(self.k)

            coeff_matrix = np.linalg.inv(coeff_matrix)
            self.M[m] = coeff_matrix.dot(coeff_vector)

    def evaluate(self, train_users, test_users):
        train_sse, test_sse, train_n, test_n = 0, 0, 0, 0
        for user in train_users:
            Ta = self.U[user.id]
            for m, r in user.movie_answers.items():
                v = self.M[m]
                train_sse += (Ta @ v - r) ** 2
                train_n += 1

        for user in test_users:
            Ta = self.U[user.id]
            for m, r in user.movie_evaluation.items():
                v = self.M[m]
                test_sse += (Ta @ v - r) ** 2
                test_n += 1

        train_rmse = np.sqrt(train_sse / train_n)
        test_rmse = np.sqrt(test_sse / test_n)
        print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

        tp_train, fp_train = 0, 0
        tp_test, fp_test = 0, 0

        for user in train_users:

            top_predictions = self.top_n_movies(user, n=20)
            for m in top_predictions:
                r = user.ask(m, False)
                if r == LIKE:
                    tp_train += 1
                else:
                    fp_train += 1

        for user in test_users:
            top_predictions = self.top_n_movies(user, n=20)
            for m in top_predictions:
                r = user.eval(m)
                if r == LIKE:
                    tp_test += 1
                else:
                    fp_test += 1

        train_p = tp_train / (tp_train + fp_train)
        test_p = tp_test / (tp_test + fp_test)

        print(f'Train Precision: {tp_train / (tp_train + fp_train)} (TP {tp_train}, FP {fp_train}), Test Precision: {tp_test / (tp_test + fp_test)} (TP {tp_test} FP {fp_test})')
        return train_p * 100, test_p * 100, train_rmse, test_rmse

    def top_n_movies(self, user, n):
        Ta = self.U[user.id]
        predictions = self.M @ Ta
        sorted_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
        return [i for i, prediction in sorted_predictions][:n]

    def fit(self, train_users, test_users):

        is_entity = self.is_entity  # If the questionable_items is entities, set is_entities to True

        n_iter = 20

        print(f'Building root node')
        tree = Node(train_users, fmf=self)
        tree.calculate_profile()  # Assign a profile to the root

        train_rmses = []
        test_rmses = []
        train_precs = []
        test_precs = []

        for i in range(n_iter):
            print(f'Beginning iteration {i}')

            # 1. Fit the tree
            self.question_items = set([i for i in range(self.n_movies)] if not self.is_entity else [i for i in range(self.n_entities)])
            tree.split(is_entity=is_entity, depth=0)

            # 2. Conduct interviews to fill out the user embeddings
            self.update_user_embeddings(train_users, test_users, tree, is_entity)

            # 3. Fit the movies
            self.update_item_embeddings(train_users)

            # 4. Evaluate
            print(f'Evaluating...')
            train_p, test_p, train_rmse, test_rmse = self.evaluate(train_users, test_users)
            train_rmses.append(train_rmse)
            test_rmses.append(test_rmse)
            train_precs.append(train_p)
            test_precs.append(test_p)

            # tree.print(self.m_idx_name_map if not self.is_entity else self.e_idx_name_map)

        import matplotlib.pyplot as plt
        plt.plot(train_rmses, color='orange', label='Train RMSE')
        plt.plot(test_rmses, color='skyblue', label='Test RMSE')
        plt.plot(train_precs, color='red', label='Train precision')
        plt.plot(test_precs, color='green', label='Test precision')
        plt.legend()
        plt.title(f'FMF asking movies' if not self.is_entity else 'FMF asking entities' + f' at {n_iter} iterations')
        plt.show()