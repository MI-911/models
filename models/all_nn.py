import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class InterviewingNeuralNetwork(nn.Module):
    def __init__(self, n_questions=3, n_items=100, hidden_size=256):
        super(InterviewingNeuralNetwork, self).__init__()

        self.n_questions = n_questions
        self.n_items = n_items
        self.input_size = n_items * 2  # Question/answer pairs for every item
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)  # 2* n_item-d answer vector
        self.fc2 = nn.Linear(self.hidden_size, self.n_items)  # n_item-d question vector

        self.fc3 = nn.Linear(self.hidden_size, self.n_items)  # 100-d ratings vector

    def forward(self, answers, interviewing=True):
        if interviewing:
            # Process the answers and generate a question
            x = tt.tanh(self.fc1(answers))
            x = tt.sigmoid(self.fc2(x))
            q = ff.one_hot(x.argmax(), num_classes=self.n_items)
            return q

        else:
            # Process the answers and generate ratings
            x = tt.tanh(self.fc1(answers))
            x = tt.tanh(self.fc3(x))
            return x
