import numpy as np

# Parameters of the LSTM cell
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        # Xavier initialization
        self.wg = np.random.randn(mem_cell_ct, concat_len) / np.sqrt(concat_len)
        self.wi = np.random.randn(mem_cell_ct, concat_len) / np.sqrt(concat_len)
        self.wf = np.random.randn(mem_cell_ct, concat_len) / np.sqrt(concat_len)
        self.wo = np.random.randn(mem_cell_ct, concat_len) / np.sqrt(concat_len)

        self.bg = np.zeros(mem_cell_ct)
        self.bi = np.zeros(mem_cell_ct)
        self.bf = np.zeros(mem_cell_ct)
        self.bo = np.zeros(mem_cell_ct)

        # Gradients
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

    def apply_diff(self, lr=1.0):
        # Update weights and biases
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff

        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff

        # Reset gradients
        self.wg_diff.fill(0.)
        self.wi_diff.fill(0.)
        self.wf_diff.fill(0.)
        self.wo_diff.fill(0.)
        self.bg_diff.fill(0.)
        self.bi_diff.fill(0.)
        self.bf_diff.fill(0.)
        self.bo_diff.fill(0.)


# One node of the LSTM
class LstmNode:
    def __init__(self, lstm_param):
        self.state = None
        self.lstm_param = lstm_param

    # Forward pass will go here (gates, state, hidden output)
    # Backward pass for training will go here
    # ...


# The full LSTM network
class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []

    def x_list_add(self, x):
        # Add new input sequence step
        self.x_list.append(x)
        # Create new node here
        # ...

    def y_list_is(self, y_list, loss_layer):
        # Compute loss for sequence
        # ...
        pass

    def x_list_clear(self):
        self.x_list = []
        self.lstm_node_list = []
